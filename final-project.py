def cosine_similarity(v1, v2):
    dotprod = sum(a * b for a, b in zip(v1, v2))
    mag1 = sum(a * a for a in v1) ** 0.5
    mag2 = sum(b * b for b in v2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0
    return dotprod / (mag1 * mag2)

def find_nearest_neighbors(embs, word, top_n=10):
    if word not in embs:
        return []

    word_v = embs[word]
    similarities = {}

    for other_word, other_v in embs.items():
        if word != other_word:
            similarities[other_word] = cosine_similarity(word_v, other_v)

    # Sort words by similarity descending
    nearest = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return nearest[:top_n]