pub trait Capnp<'a> {
    type Builder;
    type Reader;

    fn write_capnp(&self, builder: &mut Self::Builder);
    fn read_capnp(reader: Self::Reader) -> Self;
}
