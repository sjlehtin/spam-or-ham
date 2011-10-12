import numpy
import sys
import math

class TreeNode(object):
    cur_id = 1

    def __init__(self):
        self.left = None
        self.right = None
        self.ratio = 0
        self.id = TreeNode.cur_id
        TreeNode.cur_id += 1

    def is_leaf(self):
        if not self.right and not self.left:
            return True

        return False

DEBUG = True
def verbose(msg):
    if DEBUG:
        sys.stderr.write("%s\n" % msg)

def split(data, m):
    assert numpy.size(data, 0) > 1
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
        return float(subcase) / num_all

    p_spam_untagged = prob_calc(num_untagged_spam, num_untagged)
    p_spam_tagged = prob_calc(num_tagged_spam, num_tagged)
    p_ham_untagged = prob_calc(num_untagged_ham, num_untagged)
    p_ham_tagged = prob_calc(num_tagged_ham, num_tagged)

    # verbose("%s %s %s %s" % (p_spam_untagged,
    #                          p_spam_tagged,
    #                          p_ham_untagged,
    #                          p_ham_tagged))

    def log_prod(pp):
        if not pp:
            return 0
        return pp * numpy.log(pp)
    impurity = - (num_tagged * (log_prod(p_spam_tagged)
                                + log_prod(p_ham_tagged))
                  + num_untagged * (log_prod(p_spam_untagged) +
                                    log_prod(p_ham_untagged)))

    return impurity/numpy.size(data, 0)

def generate_tree(data, theta, min_row_ratio):

    min_rows = min_row_ratio * numpy.size(data, 0)

    def tree_node(data):
        node = TreeNode()

        rows = numpy.size(data, 0)

        node.ratio = float(sum(data[:, 1])) / rows

        cols = numpy.size(data, 1)
        if rows < min_rows or cols < 3:
            return node

        assert rows > 1

        print ("Starting impurity calculation, number of rows %d, "
               "columns %d." %
               (rows, numpy.size(data, 1)))
        (column, min_impurity) = min([(mm, impurity(data, mm))
                                      for mm in range(2, cols)],
                                     key=lambda xx: xx[1])
        print "Min impurity: %f, column %d." % \
            (min_impurity, column)

        [tagged, untagged] = split(data, column)

        node.col = column
        if min_impurity < theta:
            return node

        node.right = tree_node(untagged)
        node.left = tree_node(tagged)

        return node

    return tree_node(data)


def dump_tree(dt):

    fp = open("tree.dot", "w")
    cur_id = 1
    nodes = {}
    def output_header():
        print >> fp, "digraph decisiontree {"

    def output_trailer():
         print >> fp, "}"

    def output_node(node):
        print >> fp, "n%s;\n" % (node.id)

    def output_edge(n1, n2):
        print >> fp, "n%s -> n%s;\n" % (n1.id, n2.id)

    def dump_node(cur):
        if cur.left:
            output_node(cur.left)
            output_edge(cur, cur.left)
            dump_node(cur.left)

        if cur.right:
            output_edge(cur, cur.right)
            output_node(cur.right)
            dump_node(cur.right)

    output_header()
    output_node(dt)
    dump_node(dt)
    output_trailer()
    fp.close()

def classify(dt, data):
    classified = numpy.zeros((numpy.size(data, 0), 3), dtype=object)

    cols = range(numpy.size(data, 1))

    def recurse(node, data, columns):
        if node.is_leaf():
            return node.ratio

        param_is_set = (data[node.col] == 1)

        data = data[:]
        columns = columns[:]
        data = numpy.delete(data, node.col)
        del columns[node.col]

        if param_is_set:
            return recurse(node.left, data, columns)
        else:
            return recurse(node.right, data, columns)

    res = [[data[row, 0], recurse(dt, data[row, :], cols)]
           for row in range(numpy.size(data, 0))]
    res = [[xx[0], (1 if xx[1] > 0.50 else 0), xx[1]] for xx in res]
    return res


if __name__ == "__main__":

    data = numpy.loadtxt("data.txt", comments="%", dtype=int,
                         converters={ 1 : lambda xx:
                                      (xx if xx != "nan" else 0)})

    training_data = data[0:1000, :]
    indices = numpy.random.permutation(numpy.size(data, 1) - 2)
    training_data[:,2:] = training_data[:,2:][:,indices]
    dt = generate_tree(training_data, 0.30, 0.05)

    dump_tree(dt)

    classified = classify(dt, data[1000:, :])

    fp = open("classified.txt", "w")
    for xx in classified:
        print >> fp, "%d %d %.4f" % (xx[0], xx[1], xx[2])
    fp.close()

    # numpy.savetxt("classified.txt", classified, fmt=["%d", "%d", "%f"])
