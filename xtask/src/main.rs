// Copyright 2025 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Makefile replacement, using `cargo xtask` pattern.

use anyhow::Result;
use clap::{Parser, Subcommand};
use xshell::{Shell, cmd};

#[derive(Parser)]
/// `chaos_theory` task runner
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run cargo-show-asm
    Asm { filter: Option<String> },
    /// Run benchmarks
    Bench {
        #[arg(long, conflicts_with = "package")]
        all: bool,

        #[arg(long)]
        build_only: bool,

        #[arg(short, long)]
        package: Option<String>,

        filter: Option<String>,
    },
    /// Run default CI checks
    Ci {
        #[arg(long, conflicts_with = "package")]
        all: bool,

        #[arg(long)]
        full: bool,

        #[arg(short, long)]
        package: Option<String>,
    },
    /// Generate Unicode tables
    GenUnicode,
    /// Run linter
    Lint {
        #[arg(long, conflicts_with = "package")]
        all: bool,

        #[arg(long)]
        nightly: bool,

        #[arg(short, long)]
        package: Option<String>,
    },
    /// Run spellchecker
    Spell,
    /// Run tests
    Test {
        #[arg(long, conflicts_with = "package")]
        all: bool,

        #[arg(short, long)]
        package: Option<String>,

        filter: Option<String>,
    },
}

#[test]
fn verify_cli() {
    use clap::CommandFactory as _;
    Cli::command().debug_assert();
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    let sh = Shell::new()?;
    let _guard = sh.push_dir(std::env::var("CARGO_WORKSPACE_DIR")?);
    match cli.command {
        Commands::Asm { filter } => run_asm(&sh, filter),
        Commands::Bench {
            all,
            build_only,
            package,
            filter,
        } => run_bench(
            &sh,
            all,
            build_only,
            &package.unwrap_or_default(),
            &filter.unwrap_or_default(),
        ),
        Commands::Ci { all, full, package } => run_ci(&sh, all, full, &package.unwrap_or_default()),
        Commands::GenUnicode => run_gen_unicode(&sh),
        Commands::Lint {
            all,
            nightly,
            package,
        } => run_lint(&sh, all, &package.unwrap_or_default(), nightly),
        Commands::Spell => run_spell(&sh),
        Commands::Test {
            all,
            package,
            filter,
        } => run_test(
            &sh,
            all,
            &package.unwrap_or_default(),
            &filter.unwrap_or_default(),
        ),
    }
}

fn root_dirs(all: bool) -> &'static [&'static str] {
    if all {
        &[
            "chaos_theory_example",
            "chaos_theory_example_fuzz",
            "chaos_theory_example_fuzz/fuzz",
            "", // last to be the last in the terminal output
        ]
    } else {
        &[""]
    }
}

fn root_dir_all_features(dir: &'static str, nightly: bool) -> Option<&'static str> {
    dir.is_empty().then_some({
        if nightly {
            "--features=all,nightly"
        } else {
            "--features=all"
        }
    })
}

fn run_asm(sh: &Shell, filter: Option<String>) -> Result<()> {
    let filter = filter.unwrap_or_default();
    cmd!(
        sh,
        "cargo asm --package chaos_theory --lib --rust -- {filter}"
    )
    .run()?;

    Ok(())
}

fn run_ci(sh: &Shell, all: bool, full: bool, package: &str) -> Result<()> {
    run_lint(sh, all, package, full)?;
    run_test(sh, all, package, "")?;
    if full {
        run_bench(sh, all, true, package, "")?;
        run_spell(sh)?;
    }

    Ok(())
}

fn run_gen_unicode(sh: &Shell) -> Result<()> {
    // By using --chars, we don't generate high-surrogate and low-surrogate codepoints,
    // but we are OK with this since in rust they can't be represented as char anyway.
    let out = cmd!(
        sh,
        "ucd-generate general-category --chars chaos_theory/UCD/"
    )
    .output()?;
    sh.write_file("chaos_theory/gen/ucd_general_category.rs", &out.stdout)?;
    let out = cmd!(
        sh,
        "ucd-generate general-category --enum --chars chaos_theory/UCD/"
    )
    .output()?;
    sh.write_file("chaos_theory/gen/ucd_general_category_enum.rs", &out.stdout)?;

    Ok(())
}

fn run_lint(sh: &Shell, all: bool, package: &str, nightly: bool) -> Result<()> {
    let package_args = if package.is_empty() {
        &[][..]
    } else {
        &["--package", package][..]
    };
    for &dir in root_dirs(all) {
        let _guard = sh.push_dir(dir);
        cmd!(sh, "cargo fmt --check {package_args...}").run()?;
        cmd!(
            sh,
            "cargo check --all-targets --no-default-features --quiet {package_args...}"
        )
        .run()?;
        let feat = root_dir_all_features(dir, false);
        cmd!(
            sh,
            "cargo clippy --all-targets {feat...} --quiet {package_args...} -- -D warnings"
        )
        .run()?;
        if nightly && dir.is_empty() {
            // `nightly` feature is only present in the main project.
            // On nightly, don't fail if there are warnings (false positives are possible).
            let feat = root_dir_all_features(dir, true);
            cmd!(
                sh,
                "cargo +nightly clippy {feat...} --quiet {package_args...}"
            )
            .run()?;
        }
    }
    // Don't lint test and bench code.
    Ok(())
}

fn run_spell(sh: &Shell) -> Result<()> {
    // Unfortunately, when offline this hangs for some time trying to check for the updates to cspell.
    cmd!(sh, "npx cspell '**'").run()?;
    Ok(())
}

fn run_test(sh: &Shell, all: bool, package: &str, filter: &str) -> Result<()> {
    let package_args = if package.is_empty() {
        &[][..]
    } else {
        &["--package", package][..]
    };
    let filter_args = if filter.is_empty() {
        &[][..]
    } else {
        &["--", filter][..]
    };
    for dir in root_dirs(all) {
        let _guard = sh.push_dir(dir);
        let feat = root_dir_all_features(dir, false);
        cmd!(
            sh,
            "cargo test {feat...} --quiet {package_args...} {filter_args...}"
        )
        .run()?;
    }
    Ok(())
}

fn run_bench(sh: &Shell, all: bool, build_only: bool, package: &str, filter: &str) -> Result<()> {
    let package_args = if package.is_empty() {
        &[][..]
    } else {
        &["--package", package][..]
    };
    for dir in root_dirs(all) {
        let feat = root_dir_all_features(dir, true);
        if build_only {
            cmd!(
                sh,
                "cargo +nightly bench {feat...} --quiet --no-run {package_args...}"
            )
            .run()?;
        } else {
            let bench_arg = if filter.is_empty() { "benches" } else { filter };
            cmd!(
                sh,
                "cargo +nightly bench {feat...} --quiet {package_args...} -- {bench_arg}"
            )
            .run()?;
        }
    }
    Ok(())
}
