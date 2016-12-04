# Homework 8
## CSCI 3202
## Will Farmer

# Purpose

The goal of this project is to use a Hidden Markov Model (abbreviated HMM) to
try to determine parts of speech in a sentence. We do this by training a model
based on a large dataset of hand-tagged sentences.

Once we have our trained model we will attempt to run several sentences through
and see if it is at all accurate.

# Procedure

The algorithm we're using is called the Viterbi Algorithm. Let's go over how
it works.

* We first initialize an empty array of dictionaries that correspond to each
  token in our given sentence. Call this array $V$.
* Now we set our first value of $V$ to be equal to the following dictionary,
  which is simply the probability that the first token (which is always `>>>`)
  is each of the known tags. This will always result in that first token being
  designated as a `START` tag because that's the way it is in our database.
```python
{
  state: {
    'prob': emissions[sentence[0]][state],
    'prev': None
  } for state in states
}
```
* We now loop over each new word in our sentence as well as each possible state
  it might be.
    * We find the transition probabilities for each state
    * Determine the max and argmax
    * And set the current index of our $V$ array to be the max probability
      multiplied by the emission probability.
* To "deconstruct a path" we then simply find the maximum probability entry in
  each item of the $V$ array. These items correspond to the "most likely" tags.

The code is as follows.

```python
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
```


# Data

We use the Penn Tree Tag database (which has been provided), which has the
following form.

```
Pierre	NNP
Vinken	NNP
,	,
61	CD
years	NNS
old	JJ
,	,
will	MD
join	VB
the	DT
board	NN
as	IN
a	DT
nonexecutive	JJ
director	NN
Nov.	NNP
29	CD
.	.

Mr.	NNP
Vinken	NNP
is	VBZ
chairman	NN
of	IN
Elsevier	NNP
N.V.	NNP
,	,
...
```

We parse this using python, and replace all newlines in-between sentences with
starting/ending tags, along with converting it to a two-dimensional array using
the `csv` module.

```python
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
```

You'll note that for performance reasons we're using a generator here instead
of a traditional parsing method. While not fully utilized in this codebase, I
find it to be a cleaner way than using `.append()`.

Once we've loaded our data into memory (only around 8MB) we then need to
determine two different probabilistic values, the Transition Probability and the
Emission Probability. These are defined as follows.

\begin{align*}
\text{Transition}
\equiv P(\text{Tag}_i | \text{Tag}_j) =
\frac{|(\text{Tag}_i \cap \text{Tag}_j)|}{|\text{Tag}_j|}\\
\text{Emission} \equiv
P(\text{Word} | \text{Tag}) =
\frac{|(\text{Word} \cap \text{Tag})|}{|\text{Tag}|}
\end{align*}

In other words, the transition probability is the probability that given some
tag, $\text{Tag}_j$, what is the probability that we transition to some other
tag, $\text{Tag}_i$.

The emission probability can also be understood similarly. It is simply the
probability that given some tag, what's the probability that a word has that
tag.

To find these results we first do some bookkeeping and get the number of tags
that exist in our database as well as a unique word set.

```python
tags = {}
words = set()
for word, tag in data:
    if tag in tags:
        tags[tag] += 1
    else:
        tags[tag] = 1
    words.add(word)
```

Now we can find the transition probabilities. We store these in a dictionary
with keys of the form $T_i \to T_j$ where each $T$ is some tag that exists in
our database. This lets us determine the probability of tags transitioning from
one to another. We divide as we go since multiplication is communicative.

```python
transitions = {'{}->{}'.format(t0, t1): 0
               for t0 in tags
               for t1 in tags}
for i in range(len(data) - 1):
    tag0, tag1 = data[i][1], data[i + 1][1]
    key = '{}->{}'.format(tag0, tag1)
    transitions[key] += 1 / tags[tag0]
```

Finding the emission probabilities is similar, except in this case we store
these probabilities in a nested dictionary structure where each unique word from
the database is associated with a dictionary of tags, each of which is
associated with the odds. This gives us a structure where we can quickly and
easily find the emission probability for any word/tag combination.

```python
emissions = {w:{t:0 for t in tags} for w in words}
for word, tag in data:
    emissions[word][tag] += 1 / tags[tag]
```

# Results

So how do we do?

Pretty well actually!

```
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]
╰─>$ ./will_farmer_hw8.py "Can you walk the walk and talk the talk ?"
>>>   | Can | you | walk | the | walk | and | talk | the | talk | ? | <<<
START | MD  | PRP | VBP  | DT  | NN   | CC  | VB   | DT  | NN   | . | END
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]
╰─>$ ./will_farmer_hw8.py "This is a sentence ."
>>>   | This | is  | a  | sentence | . | <<<
START | DT   | VBZ | DT | NN       | . | END
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]
╰─>$ ./will_farmer_hw8.py "Can a can can a can ?"
>>>   | Can | a  | can | can | a  | can | ? | <<<
START | MD  | DT | MD  | MD  | DT | MD  | . | END
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]
╰─>$ ./will_farmer_hw8.py "This might produce a result if the system works well ."
>>>   | This | might | produce | a  | result | if | the | system | works | well | . | <<<
START | DT   | MD    | VB      | DT | NN     | IN | DT  | NN     | VBZ   | RB   | . | END
```

Feel free to test yourself. To run my code, use the following syntax.

```
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]
╰─>$ ./will_farmer_hw8.py "Insert your sentence here."
```

If you get lost, run

```
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]
╰─>$ ./will_farmer_hw8.py -h
```
