#include "io.hh"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>

#if NDZIP_SUPPORT_MMAP
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif


using namespace std::string_literals;


namespace ndzip::detail {

namespace added_on {
template <typename T>
std::pair<T *, size_t> repeat_buffer_to_n_values(T *input_buffer,
                                                 const size_t input_n_values,
                                                 const size_t target_n_values) {
  T *output_buffer = new T[target_n_values];

	size_t output_buffer_offset = 0;
	size_t n_empty_values_column = target_n_values;
	while (n_empty_values_column != 0) {
		size_t n_values_to_copy = std::min(n_empty_values_column, input_n_values);
		std::memcpy(output_buffer + output_buffer_offset, input_buffer,
								n_values_to_copy * sizeof(T));
		output_buffer_offset += n_values_to_copy;
		n_empty_values_column -= n_values_to_copy;
	}

  return std::make_pair(output_buffer, target_n_values);
}

long get_remaining_file_size(FILE* file) {
    if (!file) return -1;

    long current_pos = ftell(file);  
    fseek(file, 0, SEEK_END);       
    long end_pos = ftell(file);       
    fseek(file, current_pos, SEEK_SET); 

    return end_pos - current_pos;
}

}

class stdio_input_stream final : public input_stream {
  public:
    explicit stdio_input_stream(const std::string &file_name, size_t chunk_size)
        : _chunk_size(chunk_size), _chunk(malloc(chunk_size)) {
        if (!file_name.empty() && file_name != "-") {
            _file = fopen(file_name.c_str(), "rb");
            if (!_file) { throw io_error("fopen: " + file_name + ": " + strerror(errno)); }
        } else {
            _file = freopen(nullptr, "rb", stdin);
            if (!_file) { throw io_error("freopen: stdin: "s + strerror(errno)); }
        }
    }

    ~stdio_input_stream() noexcept(false) override {
        free(_chunk);
        if (_file != stdin) {
            if (fclose(_file) != 0) { throw io_error("fclose: "s + strerror(errno)); }
        }
    }

    stdio_input_stream(const stdio_input_stream &) = delete;
    stdio_input_stream &operator=(const stdio_input_stream &) = delete;

    std::pair<const void *, size_t> read_some(size_t remainder_from_last_chunk) override {
        assert(_n_chunks > 0 || remainder_from_last_chunk == 0);
        assert(remainder_from_last_chunk <= _chunk_size);
        const size_t bytes_to_buffer = _chunk_size - remainder_from_last_chunk;
				const size_t bytes_remaining = added_on::get_remaining_file_size(_file);
				const size_t bytes_to_read = std::min(bytes_remaining, bytes_to_buffer);

        memmove(_chunk, static_cast<std::byte *>(_chunk) + remainder_from_last_chunk, remainder_from_last_chunk);

        auto bytes_read = fread(static_cast<std::byte *>(_chunk) + remainder_from_last_chunk, 1, bytes_to_read, _file);
        if (bytes_read < bytes_to_read && ferror(_file)) { throw io_error("fread: "s + strerror(errno)); }

				added_on::repeat_buffer_to_n_values<std::byte>(
						reinterpret_cast<std::byte*>(_chunk), bytes_to_read, bytes_to_buffer);

        auto bytes_in_chunk = remainder_from_last_chunk + bytes_to_buffer;
        if (bytes_in_chunk > 0) { ++_n_chunks; }
        return {_chunk, bytes_in_chunk};
    }

    const void *read_exact() override {
        auto bytes_read = read_some(0).second;
        if (bytes_read == _chunk_size) {
            return _chunk;
        } else if (bytes_read == 0) {
            return nullptr;
        } else {
            throw io_error("Input file size is not a multiple of the chunk size");
        }
    }

  private:
    FILE *_file;
    size_t _chunk_size;
    void *_chunk;
    size_t _n_chunks = 0;
};

class stdio_output_stream final : public output_stream {
  public:
    explicit stdio_output_stream(const std::string &file_name, size_t max_chunk_length)
        : _max_chunk_length(max_chunk_length), _buffer(calloc(1, max_chunk_length)) {
        if (!file_name.empty() && file_name != "-") {
            _file = fopen(file_name.c_str(), "wb");
            if (!_file) { throw io_error("fopen: " + file_name + ": " + strerror(errno)); }
        } else {
            _file = freopen(nullptr, "wb", stdin);
            if (!_file) { throw io_error("freopen: stdout: "s + strerror(errno)); }
        }
    }

    stdio_output_stream(const stdio_output_stream &) = delete;
    stdio_output_stream &operator=(const stdio_output_stream &) = delete;

    ~stdio_output_stream() noexcept(false) override {
        free(_buffer);
        if (_file != stdout) {
            if (fclose(_file) != 0) { throw io_error("fclose: "s + strerror(errno)); }
        }
    }

    void *get_write_buffer() override {
        if (_should_zero_buffer) {
            memset(_buffer, 0, _max_chunk_length);
            _should_zero_buffer = false;
        }
        return _buffer;
    }

    void commit_chunk(size_t length) override {
        _should_zero_buffer = true;
        if (fwrite(_buffer, length, 1, _file) < 1) { throw io_error("fwrite: "s + strerror(errno)); }
    }

  private:
    FILE *_file;
    size_t _max_chunk_length;
    void *_buffer;
    bool _should_zero_buffer = false;
};

#if NDZIP_SUPPORT_MMAP

class mmap_input_stream final : public input_stream {
  public:
    explicit mmap_input_stream(const std::string &file_name, size_t max_chunk_size) : _max_chunk_size(max_chunk_size) {
        if (!file_name.empty() && file_name != "-") {
            _fd = open(file_name.c_str(), O_RDONLY);
            if (_fd == -1) { throw io_error("open: " + file_name + ": " + strerror(errno)); }
        }

        struct stat buf {};
        if (fstat(_fd, &buf) == -1) {
            if (_fd != STDIN_FILENO) { close(_fd); }
            throw io_error("fstat: " + file_name + ": " + strerror(errno));
        }
        _size = static_cast<size_t>(buf.st_size);
        _map = mmap(nullptr, _size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, _fd, 0);
        if (_map == MAP_FAILED) {
            if (_fd != STDIN_FILENO) { close(_fd); }
            throw io_error("mmap: " + file_name + ": " + strerror(errno));
        }
    }

    mmap_input_stream(const mmap_input_stream &) = delete;
    mmap_input_stream &operator=(const mmap_input_stream &) = delete;

    ~mmap_input_stream() noexcept(false) override {
        auto munmap_result = munmap(_map, _size);
        auto close_result = _fd != STDIN_FILENO ? close(_fd) : 0;
        if (munmap_result == -1) { throw io_error("munmap: input: "s + strerror(errno)); }
        if (close_result == -1) { throw io_error("close: input: "s + strerror(errno)); }
    }

    std::pair<const void *, size_t> read_some(size_t remainder_from_last_chunk) override {
        assert(remainder_from_last_chunk <= std::min(_max_chunk_size, _offset));
        _offset -= remainder_from_last_chunk;
        const void *chunk_addr = static_cast<const char *>(_map) + _offset;
        size_t chunk_size = std::min(_max_chunk_size, _size - _offset);
        _offset = std::min(_size, _offset + _max_chunk_size);
        return {chunk_addr, chunk_size};
    }

    const void *read_exact() override {
        auto [chunk, bytes_read] = read_some(0);
        if (bytes_read == _max_chunk_size) {
            return chunk;
        } else if (bytes_read == 0) {
            return nullptr;
        } else {
            throw io_error("Input file size is not a multiple of the chunk size");
        }
    }

  private:
    int _fd = STDIN_FILENO;
    size_t _max_chunk_size;
    void *_map = nullptr;
    size_t _size = 0;
    size_t _offset = 0;
};

class mmap_output_stream final : public output_stream {
  public:
    explicit mmap_output_stream(const std::string &file_name, size_t max_chunk_size) : _max_chunk_size(max_chunk_size) {
        if (!file_name.empty() && file_name != "-") {
            _fd = open(file_name.c_str(), O_RDWR | O_TRUNC | O_CREAT, (mode_t) 0666);
            if (_fd == -1) { throw io_error("open: " + file_name + ": " + strerror(errno)); }
        }
    }

    mmap_output_stream(const mmap_input_stream &) = delete;
    mmap_output_stream &operator=(const mmap_input_stream &) = delete;

    ~mmap_output_stream() noexcept(false) override {
        try {
            unmap_if_mapped();
            if (_fd != STDOUT_FILENO) { close(_fd); }
        } catch (...) {
            if (_fd != STDOUT_FILENO) { close(_fd); }
            throw;
        }
    }

    void *get_write_buffer() override {
        if (!_map) {
            truncate(_size + _max_chunk_size);
            map();
        }
        return static_cast<char *>(_map) + _size;
    }

    void commit_chunk(size_t length) override {
        unmap_if_mapped();
        _size += length;
        truncate(_size);
    }

  private:
    int _fd = STDOUT_FILENO;
    size_t _max_chunk_size;
    size_t _size = 0;
    size_t _capacity = 0;
    void *_map = nullptr;

    void unmap_if_mapped() {
        if (_map) {
            if (munmap(_map, _capacity) == -1) { throw io_error("munmap: "s + strerror(errno)); }
        }
        _map = nullptr;
    }

    void truncate(size_t new_capacity) {
        unmap_if_mapped();
        if (ftruncate(_fd, new_capacity) == -1) { throw io_error("ftruncate: "s + strerror(errno)); }
        _capacity = new_capacity;
    }

    void map() {
        assert(!_map);
        auto proto_map = mmap(nullptr, _capacity, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
        if (proto_map != MAP_FAILED) {
            _map = proto_map;
        } else {
            throw io_error("mmap: "s + strerror(errno));
        }
    }
};

#endif  // NDZIP_SUPPORT_MMAP

std::unique_ptr<input_stream> stdio_io_factory::create_input_stream(
        const std::string &file_name, size_t chunk_length) const {
    return std::make_unique<stdio_input_stream>(file_name, chunk_length);
}

std::unique_ptr<output_stream> stdio_io_factory::create_output_stream(
        const std::string &file_name, size_t max_chunk_length) const {
    return std::make_unique<stdio_output_stream>(file_name, max_chunk_length);
}

#if NDZIP_SUPPORT_MMAP

std::unique_ptr<input_stream> mmap_io_factory::create_input_stream(
        const std::string &file_name, size_t chunk_length) const {
    return std::make_unique<mmap_input_stream>(file_name, chunk_length);
}

std::unique_ptr<output_stream> mmap_io_factory::create_output_stream(
        const std::string &file_name, size_t max_chunk_length) const {
    return std::make_unique<mmap_output_stream>(file_name, max_chunk_length);
}

#endif  // NDZIP_SUPPORT_MMAP

}  // namespace ndzip::detail
