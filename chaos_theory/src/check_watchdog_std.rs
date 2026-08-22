// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use alloc::{
    string::String,
    sync::{Arc, Weak},
    vec::Vec,
};
use core::time::Duration;
use std::{
    sync::{Condvar, Mutex, MutexGuard, OnceLock},
    thread,
    time::Instant,
};

const POLL_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug)]
pub(super) struct HangInfo {
    activity: &'static str,
    number: Option<usize>,
    seed: Option<u32>,
}

impl HangInfo {
    pub(super) const fn new(
        activity: &'static str,
        number: Option<usize>,
        seed: Option<u32>,
    ) -> Self {
        Self {
            activity,
            number,
            seed,
        }
    }
}

#[derive(Debug)]
pub(super) struct CheckWatchdog {
    state: Arc<CheckState>,
}

impl CheckWatchdog {
    pub(super) fn new(timeout: Duration) -> Option<Self> {
        if timeout.is_zero() {
            return None;
        }
        let watchdog = GlobalWatchdog::get()?;
        let state = Arc::new(CheckState {
            watchdog: Arc::clone(watchdog),
            thread_name: thread::current().name().map(ToOwned::to_owned),
            timeout,
            running: Mutex::new(None),
        });
        lock(&watchdog.checks).push(Arc::downgrade(&state));
        watchdog.wake.notify_one();
        Some(Self { state })
    }

    pub(super) fn start(&self, info: HangInfo) -> WatchGuard {
        *lock(&self.state.running) = Some(Running {
            started: Instant::now(),
            info,
            reported: false,
        });
        self.state.watchdog.wake.notify_one();
        WatchGuard {
            state: Arc::clone(&self.state),
        }
    }
}

#[derive(Debug)]
pub(super) struct WatchGuard {
    state: Arc<CheckState>,
}

impl Drop for WatchGuard {
    fn drop(&mut self) {
        *lock(&self.state.running) = None;
        self.state.watchdog.wake.notify_one();
    }
}

#[derive(Debug)]
struct CheckState {
    watchdog: Arc<GlobalWatchdog>,
    thread_name: Option<String>,
    timeout: Duration,
    running: Mutex<Option<Running>>,
}

#[derive(Debug)]
struct Running {
    started: Instant,
    info: HangInfo,
    reported: bool,
}

#[derive(Debug)]
struct Report {
    thread_name: Option<String>,
    elapsed: Duration,
    info: HangInfo,
}

impl Report {
    fn write(&self, stderr: &mut impl std::io::Write) {
        let activity = match self.info.number {
            Some(number) => format!("{} {number}", self.info.activity),
            None => self.info.activity.to_owned(),
        };
        if let Some(thread_name) = &self.thread_name {
            let _ = writeln!(
                stderr,
                "[chaos_theory] possible hang on thread {thread_name:?}: {activity} has been running for {:?}",
                self.elapsed
            );
        } else {
            let _ = writeln!(
                stderr,
                "[chaos_theory] possible hang: {activity} has been running for {:?}",
                self.elapsed
            );
        }
        if let Some(seed) = self.info.seed {
            let _ = writeln!(
                stderr,
                "[chaos_theory] reproduce with `CHAOS_THEORY_RNG_SEED={seed:08x} CHAOS_THEORY_CHECK_ITERS=1`"
            );
        }
    }
}

#[derive(Debug)]
struct GlobalWatchdog {
    checks: Mutex<Vec<Weak<CheckState>>>,
    wake: Condvar,
}

impl GlobalWatchdog {
    fn get() -> Option<&'static Arc<Self>> {
        static WATCHDOG: OnceLock<Option<Arc<GlobalWatchdog>>> = OnceLock::new();
        WATCHDOG
            .get_or_init(|| {
                let watchdog = Arc::new(Self {
                    checks: Mutex::new(Vec::new()),
                    wake: Condvar::new(),
                });
                let thread_watchdog = Arc::clone(&watchdog);
                match thread::Builder::new()
                    .name("chaos_theory_watchdog".into())
                    .spawn(move || thread_watchdog.run())
                {
                    Ok(_handle) => Some(watchdog),
                    Err(error) => {
                        eprintln!("[chaos_theory] failed to start hang watchdog: {error}");
                        None
                    }
                }
            })
            .as_ref()
    }

    fn run(&self) {
        let mut checks = lock(&self.checks);
        loop {
            let now = Instant::now();
            let mut reports = Vec::new();
            let mut live = false;
            let mut wait = POLL_INTERVAL;
            checks.retain(|state| {
                let Some(state) = state.upgrade() else {
                    return false;
                };
                live = true;
                if let Some(running) = &mut *lock(&state.running)
                    && !running.reported
                {
                    let elapsed = now.saturating_duration_since(running.started);
                    if elapsed >= state.timeout {
                        running.reported = true;
                        reports.push(Report {
                            thread_name: state.thread_name.clone(),
                            elapsed,
                            info: running.info,
                        });
                    } else {
                        wait = wait.min(state.timeout.saturating_sub(elapsed));
                    }
                }
                true
            });

            if !reports.is_empty() {
                drop(checks);
                let stderr = std::io::stderr();
                let mut stderr = stderr.lock();
                for report in reports {
                    report.write(&mut stderr);
                }
                checks = lock(&self.checks);
                continue;
            }

            if live {
                checks = match self.wake.wait_timeout(checks, wait) {
                    Ok((checks, _timeout)) => checks,
                    Err(error) => error.into_inner().0,
                };
            } else {
                checks = match self.wake.wait(checks) {
                    Ok(checks) => checks,
                    Err(error) => error.into_inner(),
                };
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
