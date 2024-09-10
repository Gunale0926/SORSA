nmn = [
    ["emb", "embeddings"],
    ["att", "attention"],
    ["ffn", "feed_forward"],
    ["ln0", "pre_ln"],
]


def cvn(s):
    for i in nmn:
        if i[1] not in s:
            continue
        s = s.replace(i[1], i[0])
    return s


def convert_to_rwkv(a):
    d = {}
    for i in a.keys():
        if i != "head.weight":
            o = cvn(i)[5:]
        else:
            o = i
        print(i, "->", o)
        d[o] = a[i]
    return d
