cargo run -- init
cargo run -- import fasta ../fixtures/simple.fa
cargo run -- update vcf ../fixtures/simple.vcf
cargo run -- export gfa "out.gfa" --sample foo
cargo run -- view m123 --sample foo
cat ./*.dot
