import jsonlines

new_data = []
gamma = 1.0
with jsonlines.open(
    "",
    "r",
) as reader:
    for line in reader:
        queries = line["queries"]
        ranked_queries = sorted(queries, key=lambda x: queries[x], reverse=True)
        line["queries"] = {
            query: round((gamma / (gamma + idx)) ** 0.5, 4)
            for idx, query in enumerate(ranked_queries)
        }
        new_data.append(line)

with jsonlines.open(
    "",
    "w",
) as writer:
    writer.write_all(new_data)
