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

    #[repr(C)]
    pub struct Bloom {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct Map {
        _private: [u8; 0],
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
        pub fn mlx_array_item_uint64(out: *mut u64, array: Array) -> c_int;
        pub fn mlx_default_gpu_stream_new() -> Stream;
        pub fn mlx_stream_free(stream: Stream) -> c_int;
        pub fn mlx_synchronize(stream: Stream) -> c_int;
        pub fn mccl_merge_sorted_u32(out: *mut Array, a: Array, b: Array, stream: Stream) -> c_int;
        pub fn mccl_difference_sorted_u32(
            storage: *mut Array,
            count: *mut Array,
            a: Array,
            b: Array,
            stream: Stream,
        ) -> c_int;
        pub fn mccl_copy_u32(out: *mut Array, a: Array, stream: Stream) -> c_int;
        pub fn mccl_bloom_create(
            out: *mut *mut Bloom,
            num_blocks: u32,
            seed: u64,
            stream: Stream,
        ) -> c_int;
        pub fn mccl_bloom_free(filter: *mut Bloom) -> c_int;
        pub fn mccl_bloom_clear(filter: *mut Bloom) -> c_int;
        pub fn mccl_bloom_add(filter: *mut Bloom, keys: Array) -> c_int;
        pub fn mccl_bloom_contains(out: *mut Array, filter: *mut Bloom, keys: Array) -> c_int;
        pub fn mccl_bloom_read_words(filter: *mut Bloom, out: *mut u32, words: usize) -> c_int;
        pub fn mccl_map_create(
            out: *mut *mut Map,
            capacity: u32,
            seed: u32,
            stream: Stream,
        ) -> c_int;
        pub fn mccl_map_free(map: *mut Map) -> c_int;
        pub fn mccl_map_clear(map: *mut Map) -> c_int;
        pub fn mccl_map_insert(
            map: *mut Map,
            keys: Array,
            values: Array,
            inserted: *mut u64,
            full: *mut u64,
            invalid: *mut u64,
        ) -> c_int;
        pub fn mccl_map_find(
            values: *mut Array,
            found: *mut Array,
            map: *mut Map,
            keys: Array,
        ) -> c_int;
        pub fn mccl_map_erase(map: *mut Map, keys: Array, erased: *mut u64) -> c_int;
        pub fn mccl_map_rehash(map: *mut Map, capacity: u32) -> c_int;
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
/// Merge/copy operations are lazy until eval() or to_vec(); Bloom queries are eager.
/// No raw pointers escape this API.
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

/// A thread-affine blocked Bloom filter for arbitrary u32 keys.
///
/// Each block contains eight u32 words. The seed is fixed for the filter's lifetime.
/// Positive queries are approximate (false positives are possible); a completed
/// insertion never produces false negatives until the filter is cleared.
/// Creation, insertion, clearing, and queries complete on the GPU before returning.
/// The filter retains its context and is neither Clone, Send, nor Sync.
pub struct BloomFilter {
    raw: *mut ffi::Bloom,
    num_blocks: u32,
    seed: u64,
    context: Gpu,
}

impl Drop for BloomFilter {
    fn drop(&mut self) {
        unsafe { ffi::mccl_bloom_free(self.raw) };
    }
}

impl BloomFilter {
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Inserts all keys, including duplicates, zero, and u32::MAX, and waits.
    /// Keys must belong to this filter's GPU context. Empty input is accepted.
    pub fn add(&mut self, keys: &U32Array) -> Result<()> {
        self.context.check_context(keys)?;
        native_status(
            unsafe { ffi::mccl_bloom_add(self.raw, keys.raw) },
            "insert Bloom keys",
        )
    }

    /// Clears every bit and waits for completion, preserving the seed and size.
    pub fn clear(&mut self) -> Result<()> {
        native_status(
            unsafe { ffi::mccl_bloom_clear(self.raw) },
            "clear Bloom filter",
        )
    }

    /// Returns completed u32 flags in key order: zero means absent, one means
    /// possibly present. Positive results are approximate, not proof of membership.
    /// The returned snapshot is independent of subsequent insertion or clearing.
    pub fn contains(&self, keys: &U32Array) -> Result<U32Array> {
        self.context.check_context(keys)?;
        let mut out = self.context.output(keys.len);
        native_status(
            unsafe { ffi::mccl_bloom_contains(&mut out.raw, self.raw, keys.raw) },
            "query Bloom filter",
        )?;
        Ok(out)
    }

    /// Explicitly reads the block-major eight-word-per-block bitset to the host.
    /// This allocation and host readback are not part of insertion or querying.
    pub fn words(&self) -> Result<Vec<u32>> {
        let mut words = vec![0; self.num_blocks as usize * 8];
        native_status(
            unsafe { ffi::mccl_bloom_read_words(self.raw, words.as_mut_ptr(), words.len()) },
            "read Bloom words",
        )?;
        Ok(words)
    }
}

/// A fixed-capacity, thread-affine GPU u32-to-u32 map retaining its context.
/// All operations complete before returning. Keys MAX and MAX-1 are reserved;
/// every u32 value is valid. Capacity is exact, not rounded.
pub struct HashMap {
    raw: *mut ffi::Map,
    capacity: u32,
    seed: u32,
    len: usize,
    context: Gpu,
}

/// Completed insertion counts. Partial admission is reported, not an error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InsertReport {
    /// Unique newly occupied slots (updates are not counted).
    pub inserted: usize,
    /// Input elements rejected because no slot was available.
    pub full: usize,
    /// Input elements using either reserved key.
    pub invalid: usize,
}

/// Independent completed arrays in query order. Missing values and flags are
/// zero; found flags are one, including when a present value is zero or MAX.
pub struct MapLookup {
    pub values: U32Array,
    pub found: U32Array,
}

impl Drop for HashMap {
    fn drop(&mut self) {
        unsafe { ffi::mccl_map_free(self.raw) };
    }
}

impl HashMap {
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts or updates keys. For admitted duplicates, the last input value
    /// wins. Admission order when over capacity is unspecified. Empty input is valid.
    pub fn insert(&mut self, keys: &U32Array, values: &U32Array) -> Result<InsertReport> {
        self.context.check_context(keys)?;
        self.context.check_context(values)?;
        if keys.len != values.len {
            return Err(Error("map keys and values must have equal lengths".into()));
        }
        let (mut inserted, mut full, mut invalid) = (0, 0, 0);
        native_status(
            unsafe {
                ffi::mccl_map_insert(
                    self.raw,
                    keys.raw,
                    values.raw,
                    &mut inserted,
                    &mut full,
                    &mut invalid,
                )
            },
            "insert map entries",
        )?;
        let report = InsertReport {
            inserted: inserted as usize,
            full: full as usize,
            invalid: invalid as usize,
        };
        self.len += report.inserted;
        Ok(report)
    }

    /// Returns a snapshot unaffected by later map mutations or destruction.
    /// Reserved keys always produce absent results.
    pub fn find(&self, keys: &U32Array) -> Result<MapLookup> {
        self.context.check_context(keys)?;
        let mut values = self.context.output(keys.len);
        let mut found = self.context.output(keys.len);
        native_status(
            unsafe { ffi::mccl_map_find(&mut values.raw, &mut found.raw, self.raw, keys.raw) },
            "find map entries",
        )?;
        Ok(MapLookup { values, found })
    }

    /// Removes keys and reports unique entries erased, ignoring missing and
    /// reserved keys. Maintenance needed for subsequent insertion is GPU-resident.
    pub fn erase(&mut self, keys: &U32Array) -> Result<usize> {
        self.context.check_context(keys)?;
        let mut erased = 0;
        native_status(
            unsafe { ffi::mccl_map_erase(self.raw, keys.raw, &mut erased) },
            "erase map entries",
        )?;
        self.len -= erased as usize;
        Ok(erased as usize)
    }

    pub fn clear(&mut self) -> Result<()> {
        native_status(unsafe { ffi::mccl_map_clear(self.raw) }, "clear map")?;
        self.len = 0;
        Ok(())
    }

    /// Rebuilds at exactly this capacity, preserving entries and seed.
    /// Capacity smaller than len is rejected without changing the map.
    pub fn rehash(&mut self, capacity: u32) -> Result<()> {
        if capacity == 0 || capacity > c_int::MAX as u32 || (capacity as usize) < self.len {
            return Err(Error("map capacity must be in len.max(1)..=INT_MAX".into()));
        }
        native_status(
            unsafe { ffi::mccl_map_rehash(self.raw, capacity) },
            "rehash map",
        )?;
        self.capacity = capacity;
        Ok(())
    }
}

/// A lazy stable sorted multiset difference, retaining its GPU context.
///
/// Only the first `count()` elements of its private capacity buffer are initialized.
/// No API exposes the uninitialized tail. Neither this type nor its context is
/// Send or Sync.
pub struct Difference {
    storage: ffi::Array,
    count: ffi::Array,
    capacity: usize,
    stream: Rc<StreamInner>,
}

impl Drop for Difference {
    fn drop(&mut self) {
        unsafe {
            ffi::mlx_array_free(self.storage);
            ffi::mlx_array_free(self.count);
        }
    }
}

impl Difference {
    /// The storage capacity, equal to the length of A; does not evaluate the graph.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Evaluates both storage and count and synchronizes the GPU stream.
    /// This completes the full graph without reading count onto the host.
    pub fn eval(&self) -> Result<()> {
        clear_mlx_error();
        mlx_status(
            unsafe { ffi::mlx_array_eval(self.storage) },
            "evaluate difference storage",
        )?;
        clear_mlx_error();
        mlx_status(
            unsafe { ffi::mlx_array_eval(self.count) },
            "evaluate difference count",
        )?;
        clear_mlx_error();
        mlx_status(
            unsafe { ffi::mlx_synchronize(self.stream.0) },
            "synchronize GPU stream",
        )
    }

    /// Evaluates and synchronizes, then explicitly reads the scalar count to host.
    pub fn count(&self) -> Result<usize> {
        self.eval()?;
        let mut count = 0_u64;
        clear_mlx_error();
        mlx_status(
            unsafe { ffi::mlx_array_item_uint64(&mut count, self.count) },
            "read difference count",
        )?;
        let count =
            usize::try_from(count).map_err(|_| Error("difference count exceeds usize".into()))?;
        if count > self.capacity {
            return Err(Error("difference count exceeds storage capacity".into()));
        }
        Ok(count)
    }

    /// Evaluates and synchronizes, then copies only the initialized survivor prefix.
    /// Reading the count and the values both explicitly transfers data to the host.
    pub fn to_vec(&self) -> Result<Vec<u32>> {
        let count = self.count()?;
        if count == 0 {
            return Ok(Vec::new());
        }
        clear_mlx_error();
        let data = unsafe { ffi::mlx_array_data_uint32(self.storage) };
        if data.is_null() {
            return Err(mlx_failure("read difference storage"));
        }
        // The native scatter initializes exactly this contiguous prefix.
        Ok(unsafe { std::slice::from_raw_parts(data, count) }.to_vec())
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

    /// Creates a cleared, completed blocked Bloom filter with a fixed hash seed.
    /// `num_blocks` must be positive and eight times this value must fit INT_MAX.
    /// Positive membership results are approximate.
    pub fn bloom(&self, num_blocks: u32, seed: u64) -> Result<BloomFilter> {
        if num_blocks == 0 || num_blocks > c_int::MAX as u32 / 8 {
            return Err(Error("Bloom blocks must be in 1..=INT_MAX/8".into()));
        }
        let mut filter = BloomFilter {
            raw: ptr::null_mut(),
            num_blocks,
            seed,
            context: self.clone(),
        };
        native_status(
            unsafe { ffi::mccl_bloom_create(&mut filter.raw, num_blocks, seed, self.stream.0) },
            "create Bloom filter",
        )?;
        Ok(filter)
    }

    /// Creates a completed empty map with exact positive capacity <= INT_MAX.
    /// Native allocation limits may further restrict capacity.
    pub fn hash_map(&self, capacity: u32, seed: u32) -> Result<HashMap> {
        if capacity == 0 || capacity > c_int::MAX as u32 {
            return Err(Error("map capacity must be in 1..=INT_MAX".into()));
        }
        let mut map = HashMap {
            raw: ptr::null_mut(),
            capacity,
            seed,
            len: 0,
            context: self.clone(),
        };
        native_status(
            unsafe { ffi::mccl_map_create(&mut map.raw, capacity, seed, self.stream.0) },
            "create map",
        )?;
        Ok(map)
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

    /// Builds a lazy stable multiset difference A minus B.
    ///
    /// Both inputs must be sorted ascending. Each B occurrence cancels the earliest
    /// remaining equal occurrence in A; survivors retain their original order.
    /// Sortedness is not checked on the host. Unsorted inputs have unspecified
    /// results but remain memory-safe. Each input length must fit INT_MAX.
    /// Graph construction never reads the survivor count onto the host.
    pub fn difference(&self, a: &U32Array, b: &U32Array) -> Result<Difference> {
        self.check_context(a)?;
        self.check_context(b)?;
        let mut out = Difference {
            storage: unsafe { ffi::mlx_array_new() },
            count: unsafe { ffi::mlx_array_new() },
            capacity: a.len,
            stream: Rc::clone(&self.stream),
        };
        native_status(
            unsafe {
                ffi::mccl_difference_sorted_u32(
                    &mut out.storage,
                    &mut out.count,
                    a.raw,
                    b.raw,
                    self.stream.0,
                )
            },
            "difference sorted u32",
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
