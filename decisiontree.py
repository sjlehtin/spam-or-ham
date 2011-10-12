import numpy
import sys
import math

class TreeNode(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.ratio = 0

DEBUG = False
def verbose(msg):
    if DEBUG:
        sys.stderr.write("%s\n" % msg)

def split(data, m):
    cols = data[:, m]
    tagged = data[cols == 1]
    untagged = data[cols == 0]
    tagged = numpy.delete(tagged, m, 1)
    untagged = numpy.delete(untagged, m, 1)
    return [tagged, untagged]

def impurity(data, m):

    [tagged, untagged] = split(data, m)

    num_tagged = numpy.size(tagged, 0)
    num_untagged = numpy.size(untagged, 0)

    num_tagged_spam = sum(tagged[:, 1])
    num_tagged_ham = num_tagged - num_tagged_spam

    num_untagged_spam = sum(untagged[:, 1])
    num_untagged_ham = num_untagged - num_untagged_spam

    def prob_calc(subcase, num_all):
        if not num_all:
            return 0
        return subcase / num_all

    p_spam_untagged = prob_calc(num_untagged_spam, num_untagged)
    p_spam_tagged = prob_calc(num_tagged_spam, num_tagged)
    p_ham_untagged = prob_calc(num_untagged_ham, num_untagged)
    p_ham_tagged = prob_calc(num_tagged_ham, num_tagged)

    verbose("%s %s %s %s" % (p_spam_untagged,
                             p_spam_tagged,
                             p_ham_untagged,
                             p_ham_tagged))

    impurity = - (num_tagged * (numpy.nan_to_num(p_spam_tagged *
                                            numpy.log(p_spam_tagged))
                                + numpy.nan_to_num(p_ham_tagged *
                                              numpy.log(p_ham_tagged)))
                  + num_untagged *
                  (numpy.nan_to_num(p_spam_untagged *
                                    numpy.log(p_spam_untagged))
                   + numpy.nan_to_num(p_ham_untagged *
                                      numpy.log(p_ham_untagged))))

    return impurity

def generate_tree(data, theta):

    def tree_node(data):
        node = TreeNode()

        min_impurity = min([[mm, impurity(data, mm)]
                            for mm in range(2, numpy.size(data, 1))],
                           key=lambda xx: xx[1])
        print "Min impurity: %f, column %d." % min_impurity
        [tagged, untagged] = split(data, min_impurity[0])

        node.col = min_impurity[0]
        node.ratio = numpy.size(tagged, 0)/numpy.size(data, 0)

        if max_impurity < theta:
            return node

        node.left = tree_node(tagged)
        node.right = tree_node(untagged)

        return node

    return tree_node(data)

if __name__ == "__main__":

    data = numpy.loadtxt("data.txt", comments="%", dtype=int,
                         converters={ 1 : lambda xx:
                                      (xx if xx != "nan" else 0)})

    training_data = data[0:1000, :]

    generate_tree(training_data, 0.2)
