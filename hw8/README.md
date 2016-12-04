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

# Data

# Results

```
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]─[02:57:50 PM]
╰─>$ ./will_farmer_hw8.py "Can you walk the walk and talk the talk ?"
>>>   | Can | you | walk | the | walk | and | talk | the | talk | ? | <<<
START | MD  | PRP | VBP  | DT  | NN   | CC  | VB   | DT  | NN   | . | END
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]─[02:58:22 PM]
╰─>$ ./will_farmer_hw8.py "This is a sentence ."
>>>   | This | is  | a  | sentence | . | <<<
START | DT   | VBZ | DT | NN       | . | END
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]─[02:58:50 PM]
╰─>$ ./will_farmer_hw8.py "Can a can can a can ?"
>>>   | Can | a  | can | can | a  | can | ? | <<<
START | MD  | DT | MD  | MD  | DT | MD  | . | END
┬─[william@fillory:~/Dropbox/classwork/2016b/csci3202/hw8]─[02:59:06 PM]
╰─>$ ./will_farmer_hw8.py "This might produce a result if the system works well ."
>>>   | This | might | produce | a  | result | if | the | system | works | well | . | <<<
START | DT   | MD    | VB      | DT | NN     | IN | DT  | NN     | VBZ   | RB   | . | END
```
