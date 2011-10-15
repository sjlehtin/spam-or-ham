#!/usr/bin/env python

__doc__ = """\
Output human-readable representation of the flags in each message.
"""
import numpy

if __name__ == "__main__":

    column_names = open("data.txt").readline().strip().split()
    assert(column_names[0] == "%")
    column_names[0] = "index"
    column_names = numpy.array(column_names)

    data = numpy.loadtxt("data.txt", comments="%", dtype=int,
                         converters={ 1 : lambda xx:
                                      (xx if xx != "nan" else 0)})
    for row in data:
        flags = ', '.join(column_names[2:][row[2:] == 1])
        print "%d\t%s" % (row[0], flags)
