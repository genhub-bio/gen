#[allow(clippy::all)]
pub mod gen_core_capnp {
    include!(concat!(env!("OUT_DIR"), "/gen_core_capnp.rs"));
}

#[allow(clippy::all)]
pub mod gen_models_capnp {
    include!(concat!(env!("OUT_DIR"), "/gen_models_capnp.rs"));
}

#[allow(clippy::all)]
pub mod gen_schema_capnp {
    include!(concat!(env!("OUT_DIR"), "/gen_schema_capnp.rs"));
}
