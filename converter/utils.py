# Copyright 2019 Xiaomi, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import sys
import numpy as np

def kaldi_check(condition, msg):
    if not condition:
        raise Exception(msg)

def replace_item(a, old_item, new_item):
    kaldi_check(isinstance(a, list),
                "Input should be a list.")
    if old_item not in a:
        return a
    return [new_item if x==old_item else x for x in a]

def find(x, L):
    kaldi_check(isinstance(L, list),
                "Only supports find item in a list.")
    try:
        i = L.index(x)
        return i
    except ValueError:
        return None

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def check_is_continous(nums, up=True):
    kaldi_check(isinstance(nums, list),
                "Only supports check_is_continous in a list.")
    kaldi_check(len(nums) >= 1,
                "input list should have at least one item.")
    if len(nums) == 1:
        return nums
    pre = nums[0]
    for i in range(1, len(nums)):
        if up:
            if nums[i] - pre == 1:
                pre = nums
            else:
                return False
        else:
            if pre - nums[i] == 1:
                pre = nums[i]
            else:
                return False
    return True

def fetch_origin_input(input_str):
    start_idx = input_str.find('(')
    end_idx = input_str.find(')')
    inner = input_str[start_idx + 1: end_idx]
    items = inner.split(',')
    return items[0]

def splice_continous_numbers(nums):
    kaldi_check(isinstance(nums, list),
                "Only supports splice_continous_numbers in a list.")
    kaldi_check(len(nums) >= 1,
                "input list should have at least one item.")
    if len(nums) == 1:
        return nums
    new_nums = list()
    first = nums[0]
    pre = nums[0]
    new_nums.append([first])
    index = 0
    for i in range(1, len(nums)):
        if nums[i] - pre == 1:
            new_nums[index].append(nums[i])
            pre = nums[i]
        else:
            index += 1
            new_nums.append([nums[i]])
            pre = nums[i]
    return new_nums

def consume_token(token, line, pos):
    """Return line without token"""
    if token != line.split(None, 1)[0]:
        print("Unexpected token, expected '%s', got '%s'."
              % (token, line.split(None, 1)[0]))

    return line.partition(token)[2]


def read_next_token(s, pos):
    assert isinstance(s, str) and isinstance(pos, int)
    assert pos >= 0

    while pos < len(s) and s[pos].isspace():
        pos += 1
    if pos >= len(s):
        return None, pos
    initial_pos = pos
    while pos < len(s) and not s[pos].isspace():
        pos += 1
    token = s[initial_pos:pos]
    return token, pos


def read_float(line, pos, line_buffer):
    tok, pos = read_next_token(line, pos)
    f = None
    try:
        f = float(tok)
    except:
        print("{0}: at line position {1}, expected float but got {2}"
              .format(sys.argv[0], pos, tok), file=sys.stderr)
    return f, pos


def read_int(line, pos, line_buffer):
    tok, pos = read_next_token(line, pos)
    i = None
    try:
        i = int(tok)
    except:
        print("at file position %s, expected int but got %s" % (pos, tok))
    return i, pos


def read_bool(line, pos, line_buffer):
    tok, pos = read_next_token(line, pos)
    b = None
    if tok in ['F', 'False', 'false']:
        b = False
    elif tok in['T', 'True', 'true']:
        b = True
    else:
        print("at file position %s, expected bool but got %s" % (pos, tok))
    return b, pos


def read_vector(line, pos, line_buffer):
    tok, pos = read_next_token(line, pos)
    if tok != '[':
        print("{0}: at line position {1}, expected [ but got {2}"
              .format(sys.argv[0], pos, tok), file=sys.stderr)
        return None, pos
    v = []
    while True:
        tok, pos = read_next_token(line, pos)
        if tok == ']':
            break
        if tok is None:
            line = next(line_buffer)
            if line is None:
                print("encountered EOF while reading vector.")
                break
            else:
                pos = 0
                continue
        try:
            f = float(tok)
            v.append(f)
        except:
            print("{0}: at line position {1}, reading vector,"
                  " expected vector but got {2}"
                  .format(sys.argv[0], pos, tok), file=sys.stderr)
            return None, pos
    if tok is None:
        print("encountered EOF while reading vector.")
        return None, pos

    return np.array(v, dtype=np.float32), pos


def read_vector_int(line, pos, line_buffer):
    tok, pos = read_next_token(line, pos)
    if tok != '[':
        print("{0}: at line position {1}, expected [ but got {2}"
              .format(sys.argv[0], pos, tok), file=sys.stderr)
        return None, pos
    v = []
    while True:
        tok, pos = read_next_token(line, pos)
        if tok == ']':
            break
        if tok is None:
            line = next(line_buffer)
            if line is None:
                print("encountered EOF while reading vector.")
                break
            else:
                pos = 0
                continue
        try:
            i = int(tok)
            v.append(i)
        except:
            print("{0}: at line position {1}, reading vector,"
                  " expected float but got {2}"
                  .format(sys.argv[0], pos, tok), file=sys.stderr)
            return None, pos
    if tok is None:
        print("encountered EOF while reading vector.")
        return None, pos

    return np.array(v, dtype=np.int), pos


def check_for_newline(s, pos):
    assert isinstance(s, str) and isinstance(pos, int)
    assert pos >= 0
    saw_newline = False
    while pos < len(s) and s[pos].isspace():
        if s[pos] == "\n":
            saw_newline = True
        pos += 1
    return saw_newline, pos


def read_matrix(line, pos, line_buffer):
    tok, pos = read_next_token(line, pos)
    if tok != '[':
        print("{0}: at line position {1}, reading vector,"
              " expected '[' but got {2}"
              .format(sys.argv[0], pos, tok), file=sys.stderr)
        return None, pos
    # m will be an array of arrays (python arrays, not numpy arrays).
    m = []
    while True:
        # At this point, assume we're ready to read a new vector
        # (terminated by newline or by "]").
        v = []
        while True:
            tok, pos = read_next_token(line, pos)
            if tok == '[':
                tok, pos = read_next_token(line, pos)
            if tok == ']' or tok is None:
                break
            else:
                try:
                    f = float(tok)
                    v.append(f)
                except:
                    print("{0}: at line position {1}, reading vector,"
                          "expected float but got {2}"
                          .format(sys.argv[0], pos, tok), file=sys.stderr)
                    return None, pos

            saw_newline, pos = check_for_newline(line, pos)
            if saw_newline:  # Newline terminates each row of the matrix.
                break
        if len(v) > 0:
            m.append(v)
        if tok == ']':
            break
        if tok is None:
            line = next(line_buffer)
            if line is None:
                print("{0}: at line position {1}, reading vector,"
                      " expected ']' but got end of lines"
                      .format(sys.argv[0], pos), file=sys.stderr)
                break
            else:
                pos = 0

    ans_mat = None
    try:
        ans_mat = np.array(m, dtype=np.float32)
    except:
        if tok is None:
            print("{0}: at line position {1}, reading vector,"
                  " expected float but got {2}"
                  .format(sys.argv[0], pos, tok), file=sys.stderr)
    return ans_mat, pos


def is_component_type(component_type):
    return (isinstance(component_type, str) and len(component_type) >= 13 and
            component_type[0] == "<" and component_type[-10:] == "Component>")


def read_component_type(line, pos):
    component_type, pos = read_next_token(line, pos)
    if not is_component_type(component_type):
        print("{0}: error reading Component: at position {1},"
              " expected <xxxxComponent>,"
              " got: {2}".format(sys.argv[0], pos, component_type), file=sys.stderr)
        while True:
            tok, pos = read_next_token(line, pos)

            if tok is None or tok == '<ComponentName>':
                return None, pos
    return component_type, pos


def read_generic(line, pos, line_buffer, terminating_token, action_dict):
        if isinstance(terminating_token, str):
            terminating_tokens = set([terminating_token])
        else:
            terminating_tokens = terminating_token
            assert isinstance(terminating_tokens, set)
        assert isinstance(action_dict, dict)

        # d will contain the fields of the object.
        d = dict()
        orig_pos = pos
        while True:
            tok, pos = read_next_token(line, pos)
            if tok in terminating_tokens:
                break
            if tok is None:
                line = next(line_buffer)
                if line is None:
                    print("{0}: error reading object starting at position {1},"
                          "got EOF "
                          "while expecting one of: {2}"
                          .format(sys.argv[0], orig_pos, terminating_tokens),
                          file=sys.stderr)
                    break
                else:
                    pos = 0
                    continue
            if tok in action_dict:
                func, name = action_dict[tok]
                obj, pos = func(line, pos, line_buffer)
                d[name] = obj
        return d, pos


def parenthesis_split(sentence, separator=" ", lparen="(", rparen=")"):
    nb_brackets = 0
    sentence = sentence.strip(separator)

    l = [0]
    for i, c in enumerate(sentence):
        if c == lparen:
            nb_brackets += 1
        elif c == rparen:
            nb_brackets -= 1
        elif c == separator and nb_brackets == 0:
            l.append(i)
        # handle malformed string
        if nb_brackets < 0:
            raise Exception("Syntax error")

    l.append(len(sentence))
    # handle missing closing parentheses
    if nb_brackets > 0:
        raise Exception("Syntax error")

    return [sentence[i:j].strip(separator) for i, j in zip(l, l[1:])]
