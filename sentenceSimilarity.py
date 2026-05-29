from sentence_transformers import SentenceTransformer

# description = ["If you want to buy a trouble free car keep reading",
# "if you are looking for a bargain and don't have a problem with huge repair costs after buying your car please find another one."
# "Selling my 2006 BMW X5, 117K km, Difficult to find it in such low travel distances!!!",
# "Mint condition, it has had all maintenance done, recently had all gaskets on engine replaced so no oil leaks whatsoever",
# "Has had the transmission and transfer case oils replaced in the last month."]

description = ["It's a 2014 model, too old for me",
"Hi, I'm selling my 2014 Toyota Prius. It has only 65k miles. It has Backup camera, Bluetooth, CD, AUX, Keyless Go. There was no engine or transmission damage. Very clean inside, never smoke. Well maintained. Oil change every 5k miles with synthetic Toyota Original Oil. Registered until February 2019. Title on hands. It lost its clean title status due to rear bumper hit. Small scratch on the right side. I'm the second owner and had no any problem with it. The car is great, 50+ MPG. Will run 100k miles more easily. Call me or better text at any time"]

model = SentenceTransformer("bert-base-nli-mean-tokens")
sentences_vec = model.encode(description)

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([sentences_vec[0]], [sentences_vec[1]]))

