use std::env;

fn main() {
    // Only attempt to link LiteRT-LM when the `litert` feature is enabled.
    if env::var_os("CARGO_FEATURE_LITERT").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_NAME");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LINK_KIND");

    let lib_dir = env::var("LITERT_LM_LIB_DIR").unwrap_or_else(|_| {
        panic!(
            "LITERT_LM_LIB_DIR is required when building with --features litert (path to the directory containing the LiteRT-LM C API library)."
        )
    });

    let lib_name = env::var("LITERT_LM_LIB_NAME").unwrap_or_else(|_| "engine".to_string());
    let kind = env::var("LITERT_LM_LINK_KIND").unwrap_or_else(|_| "dylib".to_string());

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib={kind}={lib_name}");

    // Linking C++ archives typically requires explicitly linking the C++
    // standard library.
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") || target.contains("android") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
