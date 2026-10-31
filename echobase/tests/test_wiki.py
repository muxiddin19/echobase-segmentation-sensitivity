from echobase.wiki import chunk_fixed, chunk_paragraphs, words_and_paragraphs


def covers(chunks, n):
    return chunks[0][0] == 0 and chunks[-1][1] == n and all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))


def test_chunkers_partition_the_same_words():
    text = "\n".join(" ".join(f"p{i}w{j}" for j in range(n)) for i, n in enumerate([10, 50, 5, 30, 200, 3]))
    words, paras = words_and_paragraphs(text)
    a, b = chunk_fixed(words), chunk_paragraphs(words, paras)
    assert covers(a, len(words)) and covers(b, len(words))
    assert all(e - s >= 40 for s, e in b[:-1])
    assert all(e - s == 120 for s, e in a[:-1])
