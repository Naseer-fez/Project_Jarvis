#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <fstream>
#include <locale>
#include <codecvt>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

// Console colors on Windows/Linux
class Colors {
public:
    static void init() {
#ifdef _WIN32
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;
        if (hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
#endif
    }
    static const char* RESET;
    static const char* BOLD;
    static const char* RED;
    static const char* GREEN;
    static const char* YELLOW;
    static const char* BLUE;
    static const char* MAGENTA;
    static const char* CYAN;
};

#ifndef _WIN32
const char* Colors::RESET   = "\033[0m";
const char* Colors::BOLD    = "\033[1m";
const char* Colors::RED     = "\033[91m";
const char* Colors::GREEN   = "\033[92m";
const char* Colors::YELLOW  = "\033[93m";
const char* Colors::BLUE    = "\033[94m";
const char* Colors::MAGENTA = "\033[95m";
const char* Colors::CYAN    = "\033[96m";
#else
const char* Colors::RESET   = "\033[0m";
const char* Colors::BOLD    = "\033[1m";
const char* Colors::RED     = "\033[91m";
const char* Colors::GREEN   = "\033[92m";
const char* Colors::YELLOW  = "\033[93m";
const char* Colors::BLUE    = "\033[94m";
const char* Colors::MAGENTA = "\033[95m";
const char* Colors::CYAN    = "\033[96m";
#endif

// Options
struct SearchOptions {
    std::vector<fs::path> start_paths;
    std::string query = "";
    std::string content_query = "";
    int num_threads = 4;
    bool case_sensitive = false;
    bool no_skip = false;
    int max_results = 2000;
};

// Global counts
std::atomic<long long> files_scanned{0};
std::atomic<long long> dirs_scanned{0};
std::atomic<long long> match_count{0};
std::mutex stdout_mutex;

// Skip directories list
const std::vector<std::string> SKIP_DIR_NAMES = {
    "$recycle.bin",
    "system volume information",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "appdata",
    "winsxs",
    "servicing",
    "windows\\temp",
    "microsoft",
    "recovery"
};

bool should_skip(const fs::path& path) {
    std::string path_str = path.string();
    std::transform(path_str.begin(), path_str.end(), path_str.begin(), ::tolower);
    for (const auto& name : SKIP_DIR_NAMES) {
        if (path_str.find(name) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Convert string to lower case
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

// Check filename wildcard matching (simple glob)
bool match_pattern(const std::string& name, const std::string& pattern, bool case_sensitive) {
    if (pattern.empty()) return true;
    
    std::string n = case_sensitive ? name : to_lower(name);
    std::string p = case_sensitive ? pattern : to_lower(pattern);

    // Simple substring match if no wildcard
    if (p.find('*') == std::string::npos && p.find('?') == std::string::npos) {
        return n.find(p) != std::string::npos;
    }

    // Standard wildcard matching algorithm
    int n_len = n.length();
    int p_len = p.length();
    int i = 0, j = 0, startIndex = -1, match = 0;
    while (i < n_len) {
        if (j < p_len && (p[j] == '?' || p[j] == n[i])) {
            i++;
            j++;
        } else if (j < p_len && p[j] == '*') {
            startIndex = j;
            match = i;
            j++;
        } else if (startIndex != -1) {
            j = startIndex + 1;
            match++;
            i = match;
        } else {
            return false;
        }
    }
    while (j < p_len && p[j] == '*') {
        j++;
    }
    return j == p_len;
}

// Check if file is text/binary
bool is_likely_binary(const fs::path& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) return true;
    char buf[1024];
    in.read(buf, sizeof(buf));
    std::streamsize bytes = in.gcount();
    for (std::streamsize i = 0; i < bytes; ++i) {
        if (buf[i] == '\0') return true;
    }
    return false;
}

// Grep search inside a file
void search_file_content(const fs::path& path, const SearchOptions& opt) {
    if (opt.content_query.empty()) return;

    // Check size first, skip files larger than 50MB
    try {
        auto sz = fs::file_size(path);
        if (sz > 50 * 1024 * 1024) return;
    } catch (...) {
        return;
    }

    // Skip binary files
    if (is_likely_binary(path)) return;

    std::ifstream infile(path);
    if (!infile.is_open()) return;

    std::string line;
    int line_number = 0;
    std::string target = opt.case_sensitive ? opt.content_query : to_lower(opt.content_query);

    while (std::getline(infile, line)) {
        line_number++;
        std::string search_line = opt.case_sensitive ? line : to_lower(line);
        if (search_line.find(target) != std::string::npos) {
            long long current_matches = ++match_count;
            if (current_matches <= opt.max_results) {
                std::lock_guard<std::mutex> lock(stdout_mutex);
                std::cout << "[MATCH] " << Colors::CYAN << path.string() << Colors::RESET 
                          << ":" << Colors::YELLOW << line_number << Colors::RESET 
                          << ": " << line << "\n";
                std::cout.flush();
            }
        }
    }
}

// Thread-safe work queue
class FileSearchQueue {
    std::queue<fs::path> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> active_workers{0};
    bool done{false};
public:
    void push(const fs::path& path) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.push(path);
        }
        cv.notify_one();
    }

    bool pop(fs::path& path) {
        std::unique_lock<std::mutex> lock(mtx);
        active_workers--;
        while (queue.empty() && !done) {
            if (active_workers == 0) {
                done = true;
                cv.notify_all();
                return false;
            }
            cv.wait(lock);
        }
        if (done && queue.empty()) {
            return false;
        }
        active_workers++;
        path = std::move(queue.front());
        queue.pop();
        return true;
    }

    void add_worker() {
        active_workers++;
    }

    void set_active_workers(int count) {
        active_workers = count;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            done = true;
        }
        cv.notify_all();
    }

    bool is_done() {
        return done;
    }
};

void worker_thread(FileSearchQueue& queue, const SearchOptions& opt) {
    fs::path current_dir;
    while (queue.pop(current_dir)) {
        dirs_scanned++;
        try {
            for (const auto& entry : fs::directory_iterator(current_dir, fs::directory_options::skip_permission_denied)) {
                try {
                    if (entry.is_directory()) {
                        if (opt.no_skip || !should_skip(entry.path())) {
                            queue.push(entry.path());
                        }
                    } else if (entry.is_regular_file()) {
                        files_scanned++;
                        std::string filename = entry.path().filename().string();
                        if (match_pattern(filename, opt.query, opt.case_sensitive)) {
                            if (!opt.content_query.empty()) {
                                search_file_content(entry.path(), opt);
                            } else {
                                long long current_matches = ++match_count;
                                if (current_matches <= opt.max_results) {
                                    std::lock_guard<std::mutex> lock(stdout_mutex);
                                    std::cout << "[FILE] " << Colors::GREEN << entry.path().string() << Colors::RESET << "\n";
                                    std::cout.flush();
                                }
                            }
                        }
                    }
                } catch (...) {
                    // Ignore inaccessible single files or directories
                }
            }
        } catch (...) {
            // Ignore directory iterator exceptions
        }
    }
}

// Get all logical drives on Windows
std::vector<fs::path> get_windows_drives() {
    std::vector<fs::path> drives;
#ifdef _WIN32
    DWORD mask = GetLogicalDrives();
    for (int i = 0; i < 26; ++i) {
        if (mask & (1 << i)) {
            wchar_t drive[] = { static_cast<wchar_t>('A' + i), L':', L'\\', L'\0' };
            UINT type = GetDriveTypeW(drive);
            if (type == DRIVE_FIXED || type == DRIVE_REMOVABLE) {
                drives.push_back(fs::path(drive));
            }
        }
    }
#else
    drives.push_back("/");
#endif
    if (drives.empty()) {
        if (fs::exists("C:\\")) drives.push_back("C:\\");
        if (fs::exists("D:\\")) drives.push_back("D:\\");
    }
    return drives;
}

int main(int argc, char* argv[]) {
    Colors::init();
    SearchOptions opt;

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--path" || arg == "-p") && i + 1 < argc) {
            std::string path_val = argv[++i];
            if (path_val == "all") {
                opt.start_paths = get_windows_drives();
            } else {
                opt.start_paths.push_back(fs::path(path_val));
            }
        } else if ((arg == "--query" || arg == "-q") && i + 1 < argc) {
            opt.query = argv[++i];
        } else if ((arg == "--content" || arg == "-c") && i + 1 < argc) {
            opt.content_query = argv[++i];
        } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            opt.num_threads = std::stoi(argv[++i]);
        } else if ((arg == "--max-results" || arg == "-m") && i + 1 < argc) {
            opt.max_results = std::stoi(argv[++i]);
        } else if (arg == "--case" || arg == "-s") {
            opt.case_sensitive = true;
        } else if (arg == "--no-skip" || arg == "-n") {
            opt.no_skip = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Fast Multi-Threaded C++ File Search & Grep Tool\n\n";
            std::cout << "Usage: fast_search [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  -p, --path <path>        Path to search (or 'all' for all logical drives)\n";
            std::cout << "  -q, --query <pattern>    Filename match pattern (wildcards supported, e.g. *.py)\n";
            std::cout << "  -c, --content <text>     Text content to search for inside matched files (grep)\n";
            std::cout << "  -t, --threads <num>      Number of threads to run (default: 4)\n";
            std::cout << "  -m, --max-results <num>  Maximum number of results to display (default: 2000)\n";
            std::cout << "  -s, --case               Case sensitive search\n";
            std::cout << "  -n, --no-skip            Do not skip system/hidden folders\n";
            std::cout << "  -h, --help               Show this help message\n";
            return 0;
        }
    }

    // Default start paths
    if (opt.start_paths.empty()) {
        // Default to current directory
        opt.start_paths.push_back(fs::current_path());
    }

    // Print headers
    {
        std::lock_guard<std::mutex> lock(stdout_mutex);
        std::cout << Colors::BOLD << Colors::BLUE << "========================================\n";
        std::cout << " FAST SEARCH & GREP TOOL STARTED\n";
        std::cout << "========================================\n" << Colors::RESET;
        std::cout << "Search roots: ";
        for (const auto& p : opt.start_paths) {
            std::cout << p.string() << " ";
        }
        std::cout << "\n";
        if (!opt.query.empty()) std::cout << "Filename pattern: " << opt.query << "\n";
        if (!opt.content_query.empty()) std::cout << "Content pattern: " << opt.content_query << "\n";
        std::cout << "Threads: " << opt.num_threads << "\n\n";
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    FileSearchQueue queue;
    // Push start paths to the queue
    for (const auto& path : opt.start_paths) {
        if (fs::exists(path) && fs::is_directory(path)) {
            queue.push(path);
        } else if (fs::exists(path) && fs::is_regular_file(path)) {
            // If the start path is a file, search it immediately
            files_scanned++;
            std::string filename = path.filename().string();
            if (match_pattern(filename, opt.query, opt.case_sensitive)) {
                if (!opt.content_query.empty()) {
                    search_file_content(path, opt);
                } else {
                    std::cout << "[FILE] " << Colors::GREEN << path.string() << Colors::RESET << "\n";
                }
            }
        }
    }

    // Seed worker count
    queue.set_active_workers(opt.num_threads);

    // Start threads
    std::vector<std::thread> threads;
    for (int i = 0; i < opt.num_threads; ++i) {
        threads.emplace_back(worker_thread, std::ref(queue), std::cref(opt));
    }

    // Wait for completion
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Print summary
    {
        std::lock_guard<std::mutex> lock(stdout_mutex);
        std::cout << Colors::BOLD << Colors::BLUE << "\n========================================\n";
        std::cout << " SEARCH FINISHED\n";
        std::cout << "========================================\n" << Colors::RESET;
        std::cout << "Elapsed time   : " << Colors::YELLOW << elapsed.count() << " seconds\n" << Colors::RESET;
        std::cout << "Folders scanned: " << dirs_scanned << "\n";
        std::cout << "Files scanned  : " << files_scanned << "\n";
        std::cout << "Matches found  : " << Colors::GREEN << match_count << Colors::RESET << "\n";
        if (match_count > opt.max_results) {
            std::cout << Colors::YELLOW << "Note: Results capped at the first " << opt.max_results << " matches.\n" << Colors::RESET;
        }
    }

    return 0;
}
