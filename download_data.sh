curl https://database.nikonoel.fr/lichess_elite_2021-11.zip --output 1.zip
curl https://database.nikonoel.fr/lichess_elite_2021-10.zip --output 2.zip

unzip 1.zip
unzip 2.zip

mkdir -p pgns

paste -d '\n' lichess_elite_2021-11.pgn lichess_elite_2021-10.pgn > pgns/elite.pgn

rm lichess_elite_2021-11.pgn
rm lichess_elite_2021-10.pgn
rm 1.zip
rm 2.zip