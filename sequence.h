#ifndef ACG_SEQUENCE_H
#define ACG_SEQUENCE_H

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sequence {

struct Frame {
    std::filesystem::path snapshot_path;
    std::filesystem::path output_path;
};

namespace detail {
inline bool path_or_value_separator(std::istream::int_type c) {
    using Traits = std::istream::traits_type;
    return !Traits::eq_int_type(c, Traits::eof()) &&
           std::isspace(Traits::to_char_type(c), std::locale::classic());
}
} // namespace detail

// frames[i] corresponds to CSV frame i (zero-based, in manifest order).
// Width and height belong to the render invocation, not individual records.
// Each data line has two whitespace-separated UTF-8 paths, relative to the
// manifest's directory unless absolute. In double quotes, std::quoted uses
// backslash to escape the next character. Only whole-line comments are skipped.
inline std::vector<Frame> read_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream input(manifest_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open manifest: " + manifest_path.u8string());

    const auto directory = manifest_path.parent_path();
    std::vector<Frame> frames;
    std::size_t line_number = 0;
    std::string line, snapshot, output;
    while (std::getline(input, line)) {
        ++line_number;
        const auto first = line.find_first_not_of(" \t\r\n\v\f");
        if (first == std::string::npos || line[first] == '#') continue;

        std::istringstream row(line);
        row.imbue(std::locale::classic());
        const auto invalid = [&](const char* reason) {
            return std::runtime_error(manifest_path.u8string() + ":" +
                                      std::to_string(line_number) + ": " + reason);
        };
        if (!(row >> std::quoted(snapshot)) || snapshot.empty() || snapshot.find('\0') != std::string::npos)
            throw invalid("invalid snapshot path");
        if (!detail::path_or_value_separator(row.peek()))
            throw invalid("expected whitespace and an output path after the snapshot path");
        if (!(row >> std::quoted(output)) || output.empty() || output.find('\0') != std::string::npos)
            throw invalid("invalid or missing output path");
        row >> std::ws;
        if (!row.eof()) throw invalid("unexpected text after output path");

        frames.push_back({directory / std::filesystem::u8path(snapshot),
                          directory / std::filesystem::u8path(output)});
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("failed reading manifest: " + manifest_path.u8string());
    if (frames.empty()) throw std::runtime_error("manifest contains no frames: " + manifest_path.u8string());
    return frames;
}

// Sixteen (x, y) pairs in file order; load once per frame and pass .data()
// to the renderer. Parsing and path storage are entirely outside the ray loop.
inline std::array<float, 32> load_snapshot(const std::filesystem::path& snapshot_path) {
    std::ifstream input(snapshot_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open snapshot: " + snapshot_path.u8string());
    input.imbue(std::locale::classic());

    std::array<float, 32> values;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!(input >> values[i]) || !std::isfinite(values[i])) {
            if (input.bad()) throw std::runtime_error("failed reading snapshot: " + snapshot_path.u8string());
            throw std::runtime_error(snapshot_path.u8string() + ": expected finite float at value " +
                                     std::to_string(i + 1) + " of 32");
        }
        // Formatted extraction alone would accept e.g. "1-2" as two floats.
        if (!input.eof()) {
            const auto next = input.peek();
            if (next != std::istream::traits_type::eof() && !detail::path_or_value_separator(next))
                throw std::runtime_error(snapshot_path.u8string() + ": expected whitespace after value " +
                                         std::to_string(i + 1));
        }
    }
    input >> std::ws;
    if (input.bad()) throw std::runtime_error("failed reading snapshot: " + snapshot_path.u8string());
    if (!input.eof()) throw std::runtime_error(snapshot_path.u8string() + ": unexpected data after 32 values");
    return values;
}

} // namespace sequence

#endif // ACG_SEQUENCE_H
