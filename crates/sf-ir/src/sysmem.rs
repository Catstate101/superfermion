//! Process memory introspection (resident set size) for memory-efficiency
//! instrumentation. Pure platform reads — no extra dependencies: on Linux the
//! second field of `/proc/self/statm`, on Windows `K32GetProcessMemoryInfo`
//! from kernel32, and `0` (unavailable) everywhere else.

/// Resident set size of the current process in bytes (0 if unavailable).
#[cfg(target_os = "linux")]
pub fn rss_bytes() -> usize {
    if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(pages) = s.split_whitespace().nth(1) {
            if let Ok(p) = pages.parse::<usize>() {
                // statm reports resident pages; 4 KiB pages on every
                // configuration this crate ships for.
                return p * 4096;
            }
        }
    }
    0
}

/// Resident set size of the current process in bytes (0 if unavailable).
#[cfg(target_os = "windows")]
pub fn rss_bytes() -> usize {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    let mut c = ProcessMemoryCounters::default();
    c.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    if ok != 0 {
        c.working_set_size
    } else {
        0
    }
}

/// Resident set size of the current process in bytes (0 if unavailable).
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn rss_bytes() -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::rss_bytes;

    #[test]
    fn rss_is_plausible() {
        let r = rss_bytes();
        // Either unavailable (0) or clearly a running process (> 1 MiB).
        assert!(r == 0 || r > 1 << 20, "implausible rss: {r}");
        println!("rss = {} bytes ({:.1} MiB)", r, r as f64 / (1 << 20) as f64);
    }
}
