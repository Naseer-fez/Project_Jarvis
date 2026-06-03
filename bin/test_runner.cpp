#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <filesystem>
#include <windows.h>
#include <sstream>

namespace fs = std::filesystem;

// ANSI Terminal Color helpers
class Terminal {
public:
    static void init() {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut == INVALID_HANDLE_VALUE) return;
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }

    static std::string green(const std::string& text) { return "\x1b[32m" + text + "\x1b[0m"; }
    static std::string red(const std::string& text) { return "\x1b[31m" + text + "\x1b[0m"; }
    static std::string yellow(const std::string& text) { return "\x1b[33m" + text + "\x1b[0m"; }
    static std::string blue(const std::string& text) { return "\x1b[36m" + text + "\x1b[0m"; }
    static std::string bold(const std::string& text) { return "\x1b[1m" + text + "\x1b[0m"; }
};

// Wide-character conversion helper for Win32 API calls
std::wstring widen(const std::string& str) {
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

std::string narrow(const std::wstring& wstr) {
    if (wstr.empty()) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}

// Thread-safe Queue implementation
template<typename T>
class SafeQueue {
private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
public:
    void push(T val) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(val);
    }

    bool pop(T& val) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty()) return false;
        val = m_queue.front();
        m_queue.pop();
        return true;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }
};

// Process execution results
struct ProcessResult {
    int exitCode = -1;
    std::string output = "";
    double elapsedSeconds = 0.0;
};

ProcessResult runProcess(const std::wstring& appPath, const std::wstring& args) {
    ProcessResult result;
    
    // Create Pipe
    HANDLE hChildStd_OUT_Rd = NULL;
    HANDLE hChildStd_OUT_Wr = NULL;
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(PROCESS_INFORMATION));
    BOOL bSuccess = FALSE;

    {
        // Mutex to serialize process creation and handle closing.
        // This prevents other threads' concurrent child processes from inheriting this thread's write handle.
        static std::mutex spawnMutex;
        std::lock_guard<std::mutex> lock(spawnMutex);

        SECURITY_ATTRIBUTES saAttr;
        saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
        saAttr.bInheritHandle = TRUE;
        saAttr.lpSecurityDescriptor = NULL;

        if (!CreatePipe(&hChildStd_OUT_Rd, &hChildStd_OUT_Wr, &saAttr, 0)) {
            std::cerr << "Error: CreatePipe failed." << std::endl;
            return result;
        }

        if (!SetHandleInformation(hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0)) {
            std::cerr << "Error: SetHandleInformation failed." << std::endl;
            CloseHandle(hChildStd_OUT_Rd);
            CloseHandle(hChildStd_OUT_Wr);
            return result;
        }

        STARTUPINFOW si;
        ZeroMemory(&si, sizeof(STARTUPINFO));
        si.cb = sizeof(STARTUPINFO);
        si.hStdError = hChildStd_OUT_Wr;
        si.hStdOutput = hChildStd_OUT_Wr;
        si.dwFlags |= STARTF_USESTDHANDLES;

        // Construct command line
        std::wstring cmd = L"\"" + appPath + L"\" " + args;
        std::vector<wchar_t> cmdBuffer(cmd.begin(), cmd.end());
        cmdBuffer.push_back(0); // null terminator

        bSuccess = CreateProcessW(
            NULL,
            cmdBuffer.data(),
            NULL,          // process security attributes
            NULL,          // primary thread security attributes
            TRUE,          // handles are inherited
            0,             // creation flags
            NULL,          // use parent's environment
            NULL,          // use parent's current directory
            &si,           // STARTUPINFO pointer
            &pi            // receives PROCESS_INFORMATION
        );

        // Close the write end of the pipe immediately while holding the mutex.
        // Since we are holding the mutex, no other thread can execute CreateProcessW
        // in this split second and leak/inherit this inheritable write handle!
        CloseHandle(hChildStd_OUT_Wr);
    }

    if (!bSuccess) {
        std::wstring cmd = L"\"" + appPath + L"\" " + args;
        std::cerr << "Error: CreateProcessW failed for command: " << narrow(cmd) << std::endl;
        CloseHandle(hChildStd_OUT_Rd);
        return result;
    }

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();

    // Read output from the pipe buffer (outside the spawn mutex so other threads can start their processes!)
    DWORD dwRead;
    CHAR chBuf[4096];
    std::string outStr = "";
    while (true) {
        bSuccess = ReadFile(hChildStd_OUT_Rd, chBuf, sizeof(chBuf) - 1, &dwRead, NULL);
        if (!bSuccess || dwRead == 0) break;
        chBuf[dwRead] = '\0';
        outStr += chBuf;
    }

    // Wait until process exits
    WaitForSingleObject(pi.hProcess, INFINITE);

    auto endTime = std::chrono::high_resolution_clock::now();
    result.elapsedSeconds = std::chrono::duration<double>(endTime - startTime).count();

    DWORD dwExitCode = 0;
    if (GetExitCodeProcess(pi.hProcess, &dwExitCode)) {
        result.exitCode = static_cast<int>(dwExitCode);
    }
    result.output = outStr;

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hChildStd_OUT_Rd);

    return result;
}

// Find Python executable in the project workspace
std::wstring resolvePython() {
    std::vector<std::wstring> candidates = {
        L"jarvis_env\\Scripts\\python.exe",
        L".venv\\Scripts\\python.exe",
        L"venv\\Scripts\\python.exe"
    };
    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return fs::absolute(path).wstring();
        }
    }
    // Default to system Python
    return L"python.exe";
}

// Global stats and synchronization locks
std::mutex consoleMutex;
std::mutex statsMutex;

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    std::vector<std::wstring> failedFiles;
} stats;

// Worker thread routine
void testWorker(SafeQueue<std::wstring>& queue, const std::wstring& pythonPath, const std::wstring& extraArgs) {
    std::wstring testFile;
    while (queue.pop(testFile)) {
        std::string filenameOnly = narrow(fs::path(testFile).filename().wstring());

        {
            std::lock_guard<std::mutex> lock(consoleMutex);
            std::cout << Terminal::yellow("[RUNNING] ") << filenameOnly << std::endl;
        }

        // Construct pytest command arguments. Force disable coverage for subtests to speed up and avoid lock conflicts
        std::wstring args = L"-m pytest \"" + testFile + L"\" --no-cov";
        if (!extraArgs.empty()) {
            args += L" " + extraArgs;
        }

        ProcessResult res = runProcess(pythonPath, args);
        bool passed = (res.exitCode == 0);

        {
            std::lock_guard<std::mutex> lock(statsMutex);
            stats.total++;
            if (passed) {
                stats.passed++;
            } else {
                stats.failed++;
                stats.failedFiles.push_back(testFile);
            }
        }

        {
            std::lock_guard<std::mutex> lock(consoleMutex);
            if (passed) {
                std::cout << Terminal::green("[PASS]    ") << filenameOnly 
                          << " (" << std::fixed << std::setprecision(2) << res.elapsedSeconds << "s)" << std::endl;
            } else {
                std::cout << Terminal::red("[FAIL]    ") << filenameOnly 
                          << " (" << std::fixed << std::setprecision(2) << res.elapsedSeconds << "s)" << std::endl;
                std::cout << Terminal::bold("-------------------- OUTPUT FOR " + filenameOnly + " --------------------") << std::endl;
                std::cout << res.output;
                if (!res.output.empty() && res.output.back() != '\n') {
                    std::cout << std::endl;
                }
                std::cout << Terminal::bold("---------------------------------------------------------------------") << std::endl;
            }
        }
    }
}

void printHelp() {
    std::cout << Terminal::bold("Jarvis Ultimate Multi-Threaded C++ Test Runner") << std::endl;
    std::cout << "Usage: test_runner.exe [options] [-- pytest-args]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help           Show this help message" << std::endl;
    std::cout << "  -d, --dir <dir>      Specify test directory to scan (default: tests)" << std::endl;
    std::cout << "  -j, --jobs <count>   Set concurrent job limit (default: hardware threads count)" << std::endl;
    std::cout << std::endl;
    std::cout << "Remaining arguments are forwarded to pytest (e.g. -v, -k, etc.)" << std::endl;
}

int main(int argc, char* argv[]) {
    Terminal::init();

    std::wstring testDir = L"tests";
    int numJobs = std::thread::hardware_concurrency();
    if (numJobs == 0) numJobs = 2;

    std::wstring pytestArgs = L"";

    // Parse CLI options
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp();
            return 0;
        } else if (arg == "-d" || arg == "--dir") {
            if (i + 1 < argc) {
                testDir = widen(argv[++i]);
            } else {
                std::cerr << "Error: --dir option requires a directory path." << std::endl;
                return 1;
            }
        } else if (arg == "-j" || arg == "--jobs") {
            if (i + 1 < argc) {
                try {
                    numJobs = std::stoi(argv[++i]);
                } catch (...) {
                    std::cerr << "Error: --jobs requires an integer count." << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: --jobs option requires a count." << std::endl;
                return 1;
            }
        } else {
            // Forward other options to pytest
            if (!pytestArgs.empty()) pytestArgs += L" ";
            std::wstring warg = widen(arg);
            if (warg.find(L' ') != std::wstring::npos) {
                pytestArgs += L"\"" + warg + L"\"";
            } else {
                pytestArgs += warg;
            }
        }
    }

    // Verify test directory
    if (!fs::exists(testDir) || !fs::is_directory(testDir)) {
        std::cerr << Terminal::red("Error: test directory '" + narrow(testDir) + "' does not exist or is not a directory.") << std::endl;
        return 1;
    }

    // Locate Python
    std::wstring pythonPath = resolvePython();
    std::cout << Terminal::blue("Using Python interpreter: ") << Terminal::bold(narrow(pythonPath)) << std::endl;
    std::cout << Terminal::blue("Jobs (concurrency limit): ") << Terminal::bold(std::to_string(numJobs)) << std::endl;
    std::cout << Terminal::blue("Scanning test directory:  ") << Terminal::bold(narrow(testDir)) << std::endl;
    if (!pytestArgs.empty()) {
        std::cout << Terminal::blue("Forwarded Pytest args:   ") << Terminal::bold(narrow(pytestArgs)) << std::endl;
    }
    std::cout << std::endl;

    // Scan for test_*.py files
    SafeQueue<std::wstring> testQueue;
    int testCount = 0;
    try {
        for (const auto& entry : fs::recursive_directory_iterator(testDir)) {
            if (entry.is_regular_file()) {
                std::wstring pathStr = entry.path().wstring();
                // Exclude caches, system folders, or dependencies
                if (pathStr.find(L"__pycache__") != std::wstring::npos ||
                    pathStr.find(L".venv") != std::wstring::npos ||
                    pathStr.find(L"venv") != std::wstring::npos ||
                    pathStr.find(L"jarvis_env") != std::wstring::npos) {
                    continue;
                }
                
                std::wstring filename = entry.path().filename().wstring();
                if (filename.rfind(L"test_", 0) == 0 && filename.size() > 8 && filename.substr(filename.size() - 3) == L".py") {
                    testQueue.push(entry.path().wstring());
                    testCount++;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << Terminal::red("Error scanning files: ") << e.what() << std::endl;
        return 1;
    }

    if (testCount == 0) {
        std::cout << Terminal::yellow("No test files matching test_*.py found in directory: ") << narrow(testDir) << std::endl;
        return 0;
    }

    // Adjust job count to not spawn unnecessary threads
    if (numJobs > testCount) {
        numJobs = testCount;
    }

    // Start timer
    auto globalStartTime = std::chrono::high_resolution_clock::now();

    // Spawn Worker Threads
    std::vector<std::thread> workers;
    for (int i = 0; i < numJobs; ++i) {
        workers.push_back(std::thread(testWorker, std::ref(testQueue), std::cref(pythonPath), std::cref(pytestArgs)));
    }

    // Wait for worker threads to finish
    for (auto& w : workers) {
        if (w.joinable()) {
            w.join();
        }
    }

    auto globalEndTime = std::chrono::high_resolution_clock::now();
    double totalDuration = std::chrono::duration<double>(globalEndTime - globalStartTime).count();

    // Print summary
    std::cout << std::endl;
    std::cout << Terminal::bold("======================= SUMMARY =======================") << std::endl;
    std::cout << "Total Test Files Run: " << stats.total << std::endl;
    std::cout << "Passed:               " << Terminal::green(std::to_string(stats.passed)) << std::endl;
    std::cout << "Failed:               " << (stats.failed > 0 ? Terminal::red(std::to_string(stats.failed)) : std::to_string(stats.failed)) << std::endl;
    std::cout << "Total Elapsed Time:   " << std::fixed << std::setprecision(2) << totalDuration << " seconds" << std::endl;
    std::cout << Terminal::bold("=======================================================") << std::endl;

    if (stats.failed > 0) {
        std::cout << std::endl;
        std::cout << Terminal::red(Terminal::bold("Failed Test Files:")) << std::endl;
        for (const auto& failedFile : stats.failedFiles) {
            std::cout << "  - " << narrow(failedFile) << std::endl;
        }
        return 1;
    }

    std::cout << std::endl;
    std::cout << Terminal::green(Terminal::bold("All test files executed successfully!")) << std::endl;
    return 0;
}
