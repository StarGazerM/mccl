//! Small thread-affine wrapper around the pinned mlx-c ABI and MCCL operations.
//! Build the native libraries with CMake before building this crate.

use std::{
    cell::RefCell,
    error::Error as StdError,
    ffi::{c_char, c_int, c_void, CStr},
    fmt, ptr,
    rc::Rc,
    sync::Once,
};

mod ffi {
    use super::{c_char, c_int, c_void};

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Array {
        pub ctx: *mut c_void,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Stream {
        pub ctx: *mut c_void,
    }
    // mlx/c/array.h at the commit pinned by CMake: MLX_UINT32 = 3.
    pub const UINT32: c_int = 3;
    extern "C" {
        pub fn mlx_set_error_handler(
            handler: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
            data: *mut c_void,
            dtor: Option<unsafe extern "C" fn(*mut c_void)>,
        );
        pub fn mlx_array_new() -> Array;
        pub fn mlx_array_new_data(
            data: *const c_void,
            shape: *const c_int,
            dim: c_int,
            dtype: c_int,
        ) -> Array;
        pub fn mlx_array_free(array: Array) -> c_int;
        pub fn mlx_array_eval(array: Array) -> c_int;
        pub fn mlx_array_data_uint32(array: Array) -> *const u32;
        pub fn mlx_default_gpu_stream_new() -> Stream;
        pub fn mlx_stream_free(stream: Stream) -> c_int;
        pub fn mlx_synchronize(stream: Stream) -> c_int;
        pub fn mccl_merge_sorted_u32(out: *mut Array, a: Array, b: Array, stream: Stream) -> c_int;
        pub fn mccl_copy_u32(out: *mut Array, a: Array, stream: Stream) -> c_int;
        pub fn mccl_last_error() -> *const c_char;
    }
}

#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl StdError for Error {}
pub type Result<T, E = Error> = std::result::Result<T, E>;

thread_local! {
    static MLX_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}
static INSTALL_HANDLER: Once = Once::new();

unsafe extern "C" fn capture_mlx_error(message: *const c_char, _: *mut c_void) {
    if message.is_null() {
        return;
    }
    // Never borrow-panics or unwinds across the C ABI, including during TLS teardown.
    let _ = MLX_ERROR.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = Some(
                unsafe { CStr::from_ptr(message) }
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    });
}

fn clear_mlx_error() {
    MLX_ERROR.with(|slot| *slot.borrow_mut() = None);
}

fn mlx_failure(operation: &str) -> Error {
    let detail = MLX_ERROR.with(|slot| slot.borrow_mut().take());
    Error(format!(
        "{operation}: {}",
        detail.as_deref().unwrap_or("mlx-c failed")
    ))
}

fn mlx_status(status: c_int, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(mlx_failure(operation))
    }
}

fn native_status(status: c_int, operation: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { ffi::mccl_last_error() };
    let detail = if message.is_null() {
        "native operation failed".into()
    } else {
        unsafe { CStr::from_ptr(message) }.to_string_lossy()
    };
    Err(Error(format!("{operation}: {detail}")))
}

struct StreamInner(ffi::Stream);
impl Drop for StreamInner {
    fn drop(&mut self) {
        // Destructors cannot return errors. All explicit completion goes through eval().
        unsafe { ffi::mlx_stream_free(self.0) };
    }
}

/// GPU stream context. Neither this context nor its arrays implement Send or Sync.
///
/// The first context installs a process-wide mlx-c error handler that captures
/// error messages on the calling thread instead of the default terminating handler.
/// Applications using mlx-c directly should not replace that handler afterward.
#[derive(Clone)]
pub struct Gpu {
    stream: Rc<StreamInner>,
}

/// An owned one-dimensional MLX u32 array, retaining its stream context.
/// Operations are lazy until eval() or to_vec(). No raw pointers escape this API.
pub struct U32Array {
    raw: ffi::Array,
    len: usize,
    stream: Rc<StreamInner>,
}

impl Drop for U32Array {
    fn drop(&mut self) {
        unsafe { ffi::mlx_array_free(self.raw) };
    }
}

impl Gpu {
    pub fn new() -> Result<Self> {
        INSTALL_HANDLER.call_once(|| unsafe {
            ffi::mlx_set_error_handler(Some(capture_mlx_error), ptr::null_mut(), None);
        });
        clear_mlx_error();
        let raw = unsafe { ffi::mlx_default_gpu_stream_new() };
        if raw.ctx.is_null() {
            return Err(mlx_failure("create GPU stream"));
        }
        Ok(Self {
            stream: Rc::new(StreamInner(raw)),
        })
    }

    /// Copies host values into a freshly dispatched GPU buffer and waits for completion.
    /// Input upload and initialization are therefore easy to exclude from timings.
    pub fn upload(&self, values: &[u32]) -> Result<U32Array> {
        let shape = c_int::try_from(values.len())
            .map_err(|_| Error("array length exceeds INT_MAX".into()))?;
        clear_mlx_error();
        // The official constructor copies the data; values need not outlive this call.
        let raw =
            unsafe { ffi::mlx_array_new_data(values.as_ptr().cast(), &shape, 1, ffi::UINT32) };
        if raw.ctx.is_null() {
            return Err(mlx_failure("create u32 array"));
        }
        let host = U32Array {
            raw,
            len: values.len(),
            stream: Rc::clone(&self.stream),
        };
        let device = self.copy(&host)?;
        device.eval()?;
        Ok(device)
    }

    fn check_context(&self, array: &U32Array) -> Result<()> {
        if !Rc::ptr_eq(&self.stream, &array.stream) {
            return Err(Error("array belongs to a different GPU context".into()));
        }
        Ok(())
    }

    fn output(&self, len: usize) -> U32Array {
        // mlx_array_new is the official null/empty output handle, not an allocated array.
        U32Array {
            raw: unsafe { ffi::mlx_array_new() },
            len,
            stream: Rc::clone(&self.stream),
        }
    }

    /// Builds a lazy stable ascending merge (A before B for equal keys).
    ///
    /// Both inputs must already be sorted ascending. Sortedness is not scanned on
    /// the host; unsorted inputs have unspecified output, not memory-unsafe behavior.
    /// The combined length must fit INT_MAX.
    pub fn merge(&self, a: &U32Array, b: &U32Array) -> Result<U32Array> {
        self.check_context(a)?;
        self.check_context(b)?;
        let len = a
            .len
            .checked_add(b.len)
            .filter(|&n| n <= c_int::MAX as usize)
            .ok_or_else(|| Error("combined array length exceeds INT_MAX".into()))?;
        let mut out = self.output(len);
        native_status(
            unsafe { ffi::mccl_merge_sorted_u32(&mut out.raw, a.raw, b.raw, self.stream.0) },
            "merge sorted u32",
        )?;
        Ok(out)
    }

    /// Builds a new lazy GPU read/write dispatch, never an alias or identity copy.
    pub fn copy(&self, input: &U32Array) -> Result<U32Array> {
        self.check_context(input)?;
        let mut out = self.output(input.len);
        native_status(
            unsafe { ffi::mccl_copy_u32(&mut out.raw, input.raw, self.stream.0) },
            "forced u32 copy",
        )?;
        Ok(out)
    }
}

impl U32Array {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Evaluates this graph and synchronizes its GPU stream before returning.
    pub fn eval(&self) -> Result<()> {
        clear_mlx_error();
        mlx_status(unsafe { ffi::mlx_array_eval(self.raw) }, "evaluate array")?;
        clear_mlx_error();
        mlx_status(
            unsafe { ffi::mlx_synchronize(self.stream.0) },
            "synchronize GPU stream",
        )
    }

    /// Evaluates, waits, then copies the complete contiguous output to host memory.
    pub fn to_vec(&self) -> Result<Vec<u32>> {
        self.eval()?;
        if self.is_empty() {
            return Ok(Vec::new());
        }
        clear_mlx_error();
        let data = unsafe { ffi::mlx_array_data_uint32(self.raw) };
        if data.is_null() {
            return Err(mlx_failure("read u32 array"));
        }
        // Every constructor in this API produces a contiguous 1D u32 array of len.
        Ok(unsafe { std::slice::from_raw_parts(data, self.len) }.to_vec())
    }
}
