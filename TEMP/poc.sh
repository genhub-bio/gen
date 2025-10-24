cargo run -- init
cargo run -- import fasta ../fixtures/simple.fa
cargo run -- update vcf ../fixtures/simple.vcf
cargo run -- view m123 --sample foo
cat ./*.dot
