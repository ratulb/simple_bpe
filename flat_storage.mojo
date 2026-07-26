"""Flat contiguous byte storage for BPE token decoding.

All token bytes live in a single contiguous buffer.  Decode uses memcpy
chains with a pre-allocated output buffer for maximum throughput.
"""

from std.memory import alloc, memcpy


struct FlatTokenStorage(Sized & Movable):
    """Token byte storage indexed by ID using a single contiguous buffer.

    Layout:
      _buffer: [token_0_bytes][token_1_bytes]...[token_N_bytes]
      _capacity: allocated byte count in _buffer
      _used: bytes written so far
      _offsets[i]: start offset of token i in _buffer
      _lengths[i]: byte count of token i
    """

    var _buffer: UnsafePointer[UInt8, MutAnyOrigin]
    var _capacity: Int
    var _used: Int
    var _offsets: List[Int]
    var _lengths: List[Int]

    comptime INITIAL_CAPACITY: Int = 4096

    def __init__(out self):
        self._capacity = Self.INITIAL_CAPACITY
        self._buffer = alloc[UInt8](self._capacity)
        self._used = 0
        self._offsets = List[Int]()
        self._lengths = List[Int]()

    def __init__(out self, *, copy: Self):
        self._capacity = copy._used
        self._buffer = alloc[UInt8](self._capacity)
        memcpy(dest=self._buffer, src=copy._buffer, count=copy._used)
        self._used = copy._used
        self._offsets = copy._offsets.copy()
        self._lengths = copy._lengths.copy()

    def __init__(out self, *, deinit move: Self):
        self._buffer = move._buffer
        self._capacity = move._capacity
        self._used = move._used
        self._offsets = move._offsets^
        self._lengths = move._lengths^

    def __del__(deinit self):
        if self._capacity > 0:
            self._buffer.free()

    def __len__(self) -> Int:
        return len(self._offsets)

    def _grow(mut self, min_cap: Int):
        var new_cap = self._capacity * 2
        while new_cap < min_cap:
            new_cap *= 2
        var new_buf = alloc[UInt8](new_cap)
        memcpy(dest=new_buf, src=self._buffer, count=self._used)
        self._buffer.free()
        self._buffer = new_buf
        self._capacity = new_cap

    def _ensure_id(mut self, id: Int):
        while len(self._offsets) <= id:
            self._offsets.append(-1)
            self._lengths.append(0)

    def ensure_capacity(mut self, id: Int):
        """Grow the offsets/lengths arrays so index `id` is valid."""
        self._ensure_id(id)

    @always_inline
    def set(mut self, id: Int, bytes: Span[UInt8, _]):
        """Store bytes for a token ID by appending to the contiguous buffer."""
        var n = len(bytes)
        self._ensure_id(id)
        var needed = self._used + n
        if needed > self._capacity:
            self._grow(needed)
        for i in range(n):
            self._buffer[self._used + i] = bytes[i]
        self._offsets[id] = self._used
        self._lengths[id] = n
        self._used += n

    @always_inline
    def set(mut self, id: Int, bytes: List[UInt8]):
        self.set(id, Span[UInt8](bytes))

    def has(self, id: Int) -> Bool:
        """Check if a token ID has bytes stored."""
        return id >= 0 and id < len(self._offsets) and self._lengths[id] > 0

    def get(self, id: Int) -> List[UInt8]:
        """Get a copy of the stored bytes for a token ID."""
        if id < 0 or id >= len(self._offsets) or self._lengths[id] <= 0:
            return List[UInt8]()
        var off = self._offsets[id]
        var n = self._lengths[id]
        var result = List[UInt8](capacity=n)
        for i in range(n):
            result.append(self._buffer[off + i])
        return result^

    @always_inline
    def buffer_ptr(self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """Expose the underlying contiguous buffer for direct reads."""
        return self._buffer



    @always_inline
    def token_offset(self, id: Int) -> Int:
        return self._offsets[id]

    @always_inline
    def token_length(self, id: Int) -> Int:
        return self._lengths[id]

    def decode_to_string[origin: Origin, //](ref self, ids: Span[Int, origin]) raises -> String:
        """Decode a span of token IDs into a string using raw pointer access."""
        if len(ids) == 0:
            return String("")
        var n = len(self)
        var lengths = self._lengths.unsafe_ptr()
        var offsets = self._offsets.unsafe_ptr()
        var total: Int = 0
        for id in ids:
            if id < 0 or id >= n:
                raise Error("token ID out of range: " + String(id))
            total += lengths[id]
        if total == 0:
            return String("")
        var out_buf = alloc[UInt8](total)
        var ptr = self.buffer_ptr()
        var write_offset: Int = 0
        for id in ids:
            var length = lengths[id]
            if length > 0:
                memcpy(
                    dest=out_buf + write_offset,
                    src=ptr + offsets[id],
                    count=length,
                )
                write_offset += length
        var result = String(from_utf8_lossy=Span[UInt8](ptr=out_buf, length=write_offset))
        out_buf.free()
        return result^

    def clear(mut self):
        """Remove all stored tokens."""
        if self._capacity > 0:
            self._buffer.free()
        self._capacity = Self.INITIAL_CAPACITY
        self._buffer = alloc[UInt8](self._capacity)
        self._used = 0
        self._offsets = List[Int]()
        self._lengths = List[Int]()
