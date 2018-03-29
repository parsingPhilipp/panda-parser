from hybridtree.monadic_tokens import ConstituentTerminal


def parse_tagged_sentences(path, separator="/"):
    with open(path) as file:
        for line in file:
            sentence = []
            word_pos_pairs = line.split()
            for pair in word_pos_pairs:
                form, pos = pair.rsplit(separator, 1)
                sentence.append(ConstituentTerminal(form, pos))
            yield sentence


__all__ = ["parse_tagged_sentences"]
