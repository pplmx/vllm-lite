use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let proto_file = manifest_dir.join("proto/node.proto");
    let proto_include = manifest_dir.join("proto");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(&out_dir)
        .file_descriptor_set_path(out_dir.join("vllm_descriptor.bin"))
        .compile_protos(&[&proto_file], &[&proto_include])?;

    println!("cargo:rerun-if-changed={}", proto_file.display());

    Ok(())
}
