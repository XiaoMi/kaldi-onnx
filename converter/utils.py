#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/26
"""utils."""


def kaldi_check(condition: bool, msg: str):
  """Check of condition is True and raise message.

  Args:
    condition: condition for check.
    msg: raised message if condition is False.
  """
  if condition is False:
    raise Exception(msg)
