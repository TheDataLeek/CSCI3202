#!/usr/bin/env python3.5


import sys
import argparse
import csv

from fabulous import color


TAGFILE = None
DELIM = None


def main():
    args = get_args()
    tags, words, transitions, emissions = get_tags()
    predict(args.sentence, list(tags.keys()), transitions, emissions)


def predict(sentence, states, transitions, emissions):
    """
    Uses Viterbi Algorithm

    Assuming initial prob = 1 since starts with `>>>`
    """
    # Initialize empty states
    V = [{} for _ in sentence]

    # Set the first state probability to the emission probability of that word
    # for each tag it *might* be
    V[0] = {state: {'prob': emissions[sentence[0]][state],
                    'prev': None}
            for state in states}

    # Easy lambdas to index our dictionary
    tkey = lambda t0, t1: '{}->{}'.format(t0, t1)
    trans_prob = lambda t0, t1: transitions[tkey(t0, t1)]

    # Iterate over each word (not including first, we've already done that)
    for t in range(1, len(sentence)):
        # For each possible state that it might be
        for state in states:
            # Determine the transition probabilities for the previous state to
            # the current one.
            trans_probs = [V[t - 1][prev_state]['prob'] *
                                 trans_prob(prev_state, state)
                                 for prev_state in states]
            # Determine the max transition probability
            max_trans_prob = max(trans_probs)
            # Determine its argmax to find the state
            max_state = states[argmax(trans_probs)]
            # Set our current prob to be this maxval * emission prob and set
            # previous to the previous state
            V[t][state] = {'prob': max_trans_prob *
                                    emissions[sentence[t]][state],
                           'prev': max_state}

    # Now to find the path back it's simple, just find the max values at each
    # step.
    path = [max(item,
                key=lambda k: item[k]['prob'])
            for item in V]

    # Print it special
    print_sentence(sentence, path)


def argmax(array):
    maxval = max(array)
    for i, a in enumerate(array):
        if a == maxval:
            return i


def print_sentence(sentence, path):
    """This is just a nice formatter"""
    spaces = [(l - len(w), l - len(t))
              for w, t, l in zip(sentence, path,
                  [max(len(w), len(t))
                   for w, t in zip(sentence, path)])]
    s = [[], []]
    for i, w, t in zip(range(len(sentence)), sentence, path):
        s[0].append(w + (' ' * spaces[i][0]))
        s[1].append(t + (' ' * spaces[i][1]))
    print(color.underline(' | '.join(s[0])))
    print(' | '.join(s[1]))


def get_tags():
    """
    Process tagfile

    :return tags: dict
        {tag: count}
    :return words: dict
        {word: count}
    :return transitions: dict
        {'{}->{}'.format(tag1, tag0): P(Tag0 | Tag1)}
    :return emissions: dict
        {word: {tag: P(word | tag) for tags} for words}
    """
    # Get our data
    data = list(rows())

    # first let's get each word and the count of each tag
    tags = {}
    words = set()
    for word, tag in data:
        if tag in tags:
            tags[tag] += 1
        else:
            tags[tag] = 1
        words.add(word)

    # Now let's get the transition probabilities
    transitions = {'{}->{}'.format(t0, t1): 0
                   for t0 in tags
                   for t1 in tags}
    for i in range(len(data) - 1):
        tag0, tag1 = data[i][1], data[i + 1][1]
        key = '{}->{}'.format(tag0, tag1)
        transitions[key] += 1 / tags[tag0]

    # Test gainst provided thing
    assert((transitions['NN->DT'] * tags['NN'] - 870) < 1)

    # Now let's get the word & tag associations
    emissions = {w:{t:0 for t in tags} for w in words}
    for word, tag in data:
        emissions[word][tag] += 1 / tags[tag]

    # Moar tests
    assert((emissions['the']['DT'] * tags['DT'] - 39517) < 1)

    return tags, words, transitions, emissions


def rows():
    start = ['>>>', 'START']
    end = ['<<<', 'END']
    yield start
    with open(TAGFILE) as csvfile:
        reader = csv.reader(csvfile, delimiter=DELIM)
        for i, row in enumerate(reader):
            if len(row) > 0:
                yield row
            else:
                yield end
                yield start
    yield end


def get_args():
    global TAGFILE, DELIM

    parser = argparse.ArgumentParser(description='Part of Speech Tagger')
    parser.add_argument('-t', '--tagfile', type=str, default='penntree.tag',
                        help='Tagfile to use')
    parser.add_argument('-d', '--delim', type=str, default='\t',
                        help='Delimiter of tagfile.')
    parser.add_argument('sentence', metavar='S', type=str, nargs=1,
                        help='Input sentence')
    args = parser.parse_args()

    args.sentence = args.sentence[0]
    if args.sentence.startswith('SSSS'):
        args.sentence = args.sentence[6:]
    if args.sentence.endswith('EEEE'):
        args.sentence = args.sentence[:-5]
    if not args.sentence.startswith('>>>'):
        args.sentence = '>>> {} <<<'.format(args.sentence)
    args.sentence = ''.join(x if x not in [',', '.', '?', '!']
                                else ' ' + x
                            for x in args.sentence)
    args.sentence = args.sentence.split(' ')

    # I'm a naughty boy
    TAGFILE = args.tagfile
    DELIM   = args.delim

    return args


if __name__ == '__main__':
    sys.exit(main())
