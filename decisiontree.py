import numpy
import sys
import math

class TreeNode(object):
    cur_id = 0

    def __init__(self):
        self.left = None
        self.right = None
        self.name = ""
        self.col = -1

        self.num_tagged = 0
        self.num_elements = 0

        self.id = TreeNode.cur_id
        TreeNode.cur_id += 1


    def ratio(self):
        if not self.num_elements:
            return numpy.nan
        return self.num_tagged / float(self.num_elements)

    def is_leaf(self):
        if not self.right and not self.left:
            return True

        return False

    def is_spam(self):
        if self.ratio() > 0.5:
            return True
        return False

    def can_decide(self):
        return self.num_elements > 0

class NullTreeNode(TreeNode):
    __instance = None
    def __init__(self):
        TreeNode.__init__(self)
        self.name = "ROOT"

    @classmethod
    def get_instance(cls):
        if not cls.__instance:
            cls.__instance = NullTreeNode()
        return cls.__instance

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

    ## No data samples, useless as a split.
    #if sum(data[:, m]) == 0:
    #    return 0

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

    def log_prod(pp):
        if not pp:
            return 0
        return pp * numpy.log(pp)
    impurity = - (num_tagged * (log_prod(p_spam_tagged)
                                + log_prod(p_ham_tagged))
                  + num_untagged * (log_prod(p_spam_untagged) +
                                    log_prod(p_ham_untagged)))

    impurity = impurity/numpy.size(data, 0)
    assert impurity >= 0
    assert impurity <= 1.0
    return impurity

def generate_tree(data, column_names, theta, min_row_ratio):

    min_rows = min_row_ratio * numpy.size(data, 0)
    names = column_names[:]

    def tree_node(data, names, parent):
        node = TreeNode()
        rows = numpy.size(data, 0)

        if not rows:
            return node

        node.num_elements = rows
        node.num_tagged = sum(data[:, 1])

        cols = numpy.size(data, 1)

        if  rows < min_rows:
            print ("Stopping recursion due to too few rows (%d) untagged "
                   "after split with %s." % (rows, parent.name))
            return node

        if cols < 3:
            return node

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
        node.name = names[column]
        names = names[:]
        names = numpy.delete(names, column)

        #if numpy.size(tagged, 0) < min_rows:
        #    print ("Stopping recursion due to too few rows (%d) tagged "
        #           "after split with %s." % (numpy.size(tagged, 0),
        #                                     node.name))
        #    return node
        #
        if  min_impurity < theta:
            print ("Stopping recursion due to minimimum impurity (%.5f) "
                   "reached with %s." % (min_impurity, node.name))
            return node

        node.left = tree_node(tagged, names, node)
        node.right = tree_node(untagged, names, node)

        assert(node.left.can_decide() or node.right.can_decide())

        return node

    return tree_node(data, names, NullTreeNode.get_instance())

def dump_tree(dt):

    fp = open("classified.dot", "w")
    cur_id = 1
    nodes = {}
    def output_header():
        print >> fp, "digraph decisiontree {"

    def output_trailer():
         print >> fp, "}"

    def output_node(node):
        name = ""
        if node.col >= 0:
            name = "%s (%d)" % (node.name, node.col)
        print >> fp, "n%s [label=\"ratio=%.4f\\n%s\"];\n" % (
            node.id, node.ratio(), name)

    def output_edge(n1, n2, label=""):
        print >> fp, "n%s -> n%s [label=%s];\n" % (n1.id, n2.id, label)

    def dump_node(cur):
        if cur.left:
            output_node(cur.left)
            output_edge(cur, cur.left, label="Yes")
            dump_node(cur.left)

        if cur.right:
            output_edge(cur, cur.right, label="No")
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

    def recurse(node, parent, data, columns):
        if node.is_leaf():
            return (("ROOT" if not parent else parent.name), node)

        param_is_set = (data[node.col] == 1)

        data = data[:]
        columns = columns[:]
        data = numpy.delete(data, node.col)
        del columns[node.col]

        if node.left.can_decide() and param_is_set:
            return recurse(node.left, node, data, columns)
        else:
            return recurse(node.right, node, data, columns)

    res = [[data[row, 0], recurse(dt, None, data[row, :], cols)]
           for row in range(numpy.size(data, 0))]
    return res


if __name__ == "__main__":

    data = numpy.loadtxt("data.txt", comments="%", dtype=int,
                         converters={ 1 : lambda xx:
                                      (xx if xx != "nan" else 0)})
    column_names = open("data_column_names.txt").read().strip().split()
    column_names = numpy.array(column_names)

    validation_set_size = 100
    training_data = data[0:1000, :]
    # Shuffle columns so as not to run the algorithm with the columns
    # always in a set order.
    indices = numpy.random.permutation(numpy.size(data, 1) - 2)
    training_data[:,2:] = training_data[:,2:][:,indices]
    column_names[2:] = column_names[2:][indices]

    # Shuffle rows before separating the validation set.
    indices = numpy.random.permutation(numpy.size(training_data, 0))
    training_data = training_data[indices, :]

    validation_set = training_data[0:validation_set_size, :]
    training_data = training_data[validation_set_size:, :]

    dt = generate_tree(training_data, column_names, 0.08, 0.005)

    dump_tree(dt)

    validated = classify(dt, validation_set)

    validated = numpy.array([node.is_spam()
                             for (row, (split, node)) in validated])
    from_data = validation_set[:, 1]

    print "Validation:"
    print " Classified as spam: %d / %d" % (sum(validated),
                                            numpy.size(validated))
    print " Classified as spam in validation: %d / %d" % (sum(from_data),
                                                          numpy.size(validated))
    print " Correctness: %.3f" % (float(sum(((validated == from_data)) * 1))
                                  / validation_set_size)
    for (ii, got, exp) in zip(range(1, numpy.size(validated, 0)),
                              validated, from_data):
        if got != exp:
            print "%d: %d, %d" % (ii, got, exp)
    classified = classify(dt, data[1000:, :])

    num_spam = 0
    fp = open("classified.txt", "w")
    for (row, (last_split, node)) in classified:
        print >> fp, "%d %d %.4f # %s" % (row, node.is_spam(), node.ratio(),
                                          last_split)
        if node.is_spam():
            num_spam += 1
    print "Classified as spam: %d / %d" % (num_spam, len(classified))
    fp.close()

    # numpy.savetxt("classified.txt", classified, fmt=["%d", "%d", "%f"])
