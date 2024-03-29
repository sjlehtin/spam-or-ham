#!/usr/bin/env python

import numpy
import sys
import math
import optparse
from numpy import size # For conveniency.
import os

class TreeNode(object):
    cur_id = 0

    def __init__(self):
        self.left = None
        self.right = None
        self.name = ""
        self.col = -1

        self.num_tagged = 0
        self.num_elements = 0

        self.pruned = False

        self.id = TreeNode.cur_id
        TreeNode.cur_id += 1


    def ratio(self):
        if not self.num_elements:
            return numpy.nan
        return self.num_tagged / float(self.num_elements)

    def is_leaf(self):
        if self.pruned:
            return True

        if not self.right and not self.left:
            return True

        return False

    def is_spam(self):
        if self.ratio() > 0.5:
            return True
        return False

    def can_decide(self):
        return self.num_elements > 0

    def __str__(self):
        return self.name

DEBUG = False

def verbose(msg):
    if DEBUG:
        sys.stderr.write("%s\n" % msg)

def split(data, m):
    assert size(data, 0) > 1
    cols = data[:, m]
    tagged = data[cols == 1]
    untagged = data[cols == 0]
    tagged = numpy.delete(tagged, m, 1)
    untagged = numpy.delete(untagged, m, 1)
    return [tagged, untagged]

def impurity(data, m):

    [tagged, untagged] = split(data, m)

    num_tagged = size(tagged, 0)
    num_untagged = size(untagged, 0)

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

    impurity = impurity/size(data, 0)
    assert impurity >= 0
    assert impurity <= 1.0
    return impurity

def generate_tree(data, column_names, column_ids, theta, min_row_ratio):

    min_rows = min_row_ratio * size(data, 0)

    def tree_node(data, ids, parent):
        node = TreeNode()
        rows = size(data, 0)

        if not rows:
            return node

        node.num_elements = rows
        node.num_tagged = sum(data[:, 1])

        if  rows < min_rows:
            print ("Stopping recursion due to too few rows (%d) untagged "
                   "after split with %s." % (rows, parent.name))
            return node

        if node.num_tagged == node.num_elements:
            print ("Stopping recursion due to all elements being spam.")
            return node

        if node.num_tagged == 0:
            print ("Stopping recursion due to no element being spam.")
            return node

        # Get the columns that have meaningful data.  If a parameter does
        # not have any instances in the training set, we cannot classify
        # based on that.
        columns_to_retain = sum(data) > 1
        columns_to_retain[0:2] = True

        print "Dropping %d / %d columns." % (size(data, 1) -
                                             sum(columns_to_retain * 1),
                                             size(data, 1))

        assert(size(ids) == size(data, 1))
        data = data[:, columns_to_retain]
        ids = ids[:, columns_to_retain]

        cols = size(data, 1)

        # First 2 columns are bookkeeping, so 3 columns are needed to
        # have at least one business column.
        if cols < 3:
            print ("Stopping recursion due to too few cols (%d) "
                   "after split with %s." % (cols, parent.name))
            return node

        print ("Starting impurity calculation, number of rows %d, "
               "columns %d." %
               (rows, size(data, 1)))
        (column, min_impurity) = min([(mm, impurity(data, mm))
                                      for mm in range(2, cols)],
                                     key=lambda xx: xx[1])
        real_column = ids[column]
        print "Min impurity: %f, column %s (%d)." % \
            (min_impurity, column_names[real_column], real_column)

        [tagged, untagged] = split(data, column)

        ids = numpy.delete(ids, column)

        node.col = real_column
        node.name = column_names[real_column]

        if  min_impurity < theta:
            print ("Stopping recursion due to minimimum impurity (%.5f) "
                   "reached with %s." % (min_impurity, node))
            return node

        node.left = tree_node(tagged, ids, node)
        node.right = tree_node(untagged, ids, node)

        assert(node.left.can_decide() or node.right.can_decide())

        return node

    return tree_node(data, column_ids, None)

def dump_tree(dt, filename):
    fp = open(filename, "w")
    cur_id = 1
    nodes = {}
    def output_header():
        print >> fp, "digraph decisiontree {"

    def output_trailer():
         print >> fp, "}"

    def output_node(node):
        name = ""
        if node.col >= 0:
            name = "%s (%d)" % (node, node.col)
        print >> fp, "n%s [label=\"ratio=%.4f (%d/%d)\\n%s\"];\n" % (
            node.id, node.ratio(), node.num_tagged, node.num_elements, name)

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
    classified = numpy.zeros((size(data, 0), 3), dtype=object)

    cols = range(size(data, 1))

    def classify_row(dt, row):
        def recurse(node, parent, used_columns):
            if node.is_leaf():
                return (("ROOT" if not parent else parent.name), node)

            if node in used_columns:
                raise RuntimeError("Node %s used twice! (parent %s)" % (
                        node, parent))
            used_columns[node] = True
            param_is_set = (row[node.col] == 1)

            if node.left.can_decide() and param_is_set:
                return recurse(node.left, node, used_columns.copy())
            elif node.right.can_decide():
                return recurse(node.right, node, used_columns.copy())
            else:
                return (("ROOT" if not parent else parent.name), node)
        return recurse(dt, None, {})

    res = [[data[row, 0], classify_row(dt, data[row,:])]
           for row in range(size(data, 0))]
    return res


if __name__ == "__main__":

    parser = optparse.OptionParser(
        usage="""%prog [options] [file ..]

Generate a decision tree and validate using a training set.  Output
classified test set, which can be used to calculate generalization
error.

""")
    parser.add_option("--verbose", dest="verbose", action="store_true",
                      help="Enable verbose mode.")
    parser.add_option("--dump-tree", dest="dump_trees",
                      action="append", default=[], help="Unit-test option.")
    parser.add_option("--training-data-size", dest="training_data_size",
                      type="int",
                      default=1000, help="Training set size, chosen at "
                      "random from the given training data.")
    parser.add_option("--k-fold", dest="k", type="int",
                      default=10, help="Perform K-fold validation.  "
                      "Training data will be split K pieces, and the "
                      "classifier will be trained K times and the one "
                      "performing best against the validation set is chosen.")
    parser.add_option("--data-file", dest="data_file",
                      default="full_data.txt", help="Data source to use.")
    parser.add_option("--disable-pruning", dest="prune", action="store_false",
                      default=True, help="Disable postpruning.")
    parser.add_option("--preprune", dest="preprune", action="store_true",
                      default=False, help="Enable prepruning.")
    parser.add_option("--diff", dest="diff", action="store_true",
                      default=False, help="Output list of misclassified "
                      "entries on the test set.")
    (opts, args) = parser.parse_args()

    if opts.verbose:
        DEBUG = True

    data = numpy.loadtxt(opts.data_file, comments="%", dtype=int,
                         converters={ 1 : lambda xx:
                                      (xx if xx != "nan" else 0)})
    column_names = open("data_column_names.txt").read().strip().split()
    column_names = numpy.array(column_names)

    # Rows which have unknown spam-ham status cannot be used in
    # training.  Currently relies on full_data.txt being loaded in which
    # all the rows are correcly marked.  XXX if nan's are used to
    # indicate unknown classification (as with data.txt), numpy will
    # load the whole matrix as floats (unless a converter is used to
    # force the nan to some integer value), which isn't very convenient
    # with this type of data.
    ind = numpy.random.permutation(size(data, 0))
    training_data = data[ind[0:opts.training_data_size], :]
    data = data[ind[opts.training_data_size:],:]

    verbose("training set possible size: %s" % size(training_data, 0))
    verbose("data set possible size: %s" % size(data, 0))

    # Shuffle rows before separating the validation set.
    indices = numpy.random.permutation(size(training_data, 0))
    training_data = training_data[indices, :]

    data = numpy.append(data, training_data[opts.training_data_size:,:], axis=0)
    training_data = training_data[0:opts.training_data_size, :]

    verbose("training set size after: %s,%s" % (size(training_data, 0),
                                             size(training_data, 1)))
    verbose("data size after: %s,%s" % (size(data, 0),
                                        size(data, 1)))

    column_ids = numpy.array(range(0, size(data, 1)))

    # Shuffle columns so as not to run the algorithm with the columns
    # always in a set order.
    indices = numpy.random.permutation(size(data, 1) - 2)
    data[:,2:] = data[:,2:][:,indices]
    training_data[:,2:] = training_data[:,2:][:,indices]
    column_names[2:] = column_names[2:][indices]

    def check_results(validated, validation_data):
        nodes = [node for (row, (split, node)) in validated]
        validated = numpy.array([node.is_spam() for node in nodes])
        probabilities = numpy.array([node.ratio() for node in nodes])
        pis = numpy.zeros(len(nodes))
        from_data = validation_data[:, 1]

        pis[:] = probabilities[:]
        assert sum((probabilities > 1) * 1) == 0

        ind = numpy.isnan(probabilities)

        for ii, value in [pair for pair in enumerate(ind) if pair[1]]:
            node = nodes[ii]
            print "%s: %.3f %s" % (node, node.ratio(), node.can_decide())

        assert sum(numpy.isnan(probabilities) * 1) == 0

        classified_as_ham = (from_data == 0)
        pis[:, classified_as_ham] = \
            probabilities[:, classified_as_ham] * -1 + 1
        # Add a small value to make logarithms valid.
        pis += numpy.finfo(numpy.float).eps
        assert sum((pis < 0) * 1) == 0
        assert sum(numpy.isnan(pis) * 1) == 0

        results = {}
        results['num_samples'] = size(validated)
        results['classified_as_spam'] = sum(validated)
        results['really_spam'] = sum(from_data)
        results['correctly_classified'] = sum((validated == from_data) * 1)
        results['correctly_classified_as_spam'] = \
            sum(numpy.logical_and((validated == 1), (from_data == 1)) * 1)
        results['accuracy'] = (float(results['correctly_classified'])
                               / results['num_samples'])
        results['recall'] = (float(results['correctly_classified_as_spam'])/
                             results['really_spam'])
        results['precision'] = (float(results['correctly_classified_as_spam'])/
                                results['classified_as_spam'])
        results['perplexity'] = numpy.exp(-numpy.mean(numpy.log(pis)))

        diff = []
        for (ii, got, exp) in zip(range(1, results['num_samples']),
                                  validated, from_data):
            if got != exp:
                diff.append((ii, got, exp))
        results['differences'] = diff
        return results

    def dump_results(results, give_diff=False):
        print "Validation:"
        print " Classified as spam: %d / %d" % (results['classified_as_spam'],
                                                results['num_samples'])
        print " Classified as spam in validation set: %d / %d" % (
            results['really_spam'],
            results['num_samples'])
        print " Correctly classified:",  results['correctly_classified']
        print " Correctly classified as spam:",  \
            results['correctly_classified_as_spam']
        print " Accuracy: %.3f" % (results['accuracy'])
        print " Precision: %.3f" % (results['precision'])
        print " Recall: %.3f" % (results['recall'])
        print " Perplexity: %.3f" % (results['perplexity'])
        if give_diff:
            print "Index:\tgot,\texpected"
            if results['differences']:
                for (ii, got, exp) in results['differences']:
                    print "%d:\t%d,\t%d" % (ii, got, exp)

    def calculate_results(dt, validation_data):
        validated = classify(dt, validation_data)
        res = check_results(validated, validation_data)
        return res

    def calculate_accuracy(dt, validation_data):
        return calculate_results(dt, validation_data)['accuracy']

    def prune(tree, pruning_data):
        # The list here is a kludge to get around python scoping.
        max_accuracy = [calculate_accuracy(tree, pruning_data)]
        def prune_one(node):
            if not node:
                return

            if node.is_leaf():
                return

            node.pruned = True
            current_accuracy = calculate_accuracy(tree, pruning_data)
            verbose("Got accurace %s, maximum %s." % (current_accuracy,
                                                      max_accuracy[0]))
            if current_accuracy >= max_accuracy[0]:
                max_accuracy[0] = current_accuracy
                verbose("Pruning node %s." % node)
                node.left = None
                node.right = None
            else:
                node.pruned = False
                prune_one(node.left)
                prune_one(node.right)

        prune_one(tree)
        return tree

    def train_one(train_index, training_data, pruning_data, validation_data):
        verbose("training data size: %s" % (size(training_data, 0)))
        verbose("validation data size: %s" % (size(validation_data, 0)))
        theta = 0
        min_row_ratio = 0
        if opts.preprune:
            theta = 0.04
            min_row_ratio = 0.002
            print "Prepruning, theta:", theta, "min_row_ratio:", min_row_ratio
        dt = generate_tree(training_data, column_names, column_ids,
                           theta, min_row_ratio)
        orig = None
        pruned = None
        if 'original' in opts.dump_trees:
            orig = "classified-original-%d.dot" % train_index
            dump_tree(dt, orig)

        if opts.prune:
            dt = prune(dt, pruning_data)
            if 'pruned' in opts.dump_trees:
                pruned = "classified-pruned-%d.dot" % train_index
                dump_tree(dt, pruned)

        results = calculate_results(dt, validation_data)
        results['original'] = orig
        results['pruned'] = pruned
        return [results, dt]

    def training_set_split(training_data):
        validation_set_size = size(training_data, 0)/opts.k
        assert validation_set_size < size(training_data, 0)/2

        for ii in range(opts.k):
            validation = training_data[
                ii * validation_set_size :
                    ii * validation_set_size + validation_set_size, :]
            train = numpy.append(
                training_data[0:ii * validation_set_size, :],
                training_data[ii * validation_set_size + validation_set_size:,
                              :], axis=0)
            ind = numpy.random.permutation(size(train, 0))
            prune = train[ind[0:validation_set_size], :]
            train = train[ind[validation_set_size:], :]

            yield (train, prune, validation)

    trees = [train_one(ii, *train)
             for (ii, train) in
             enumerate([train
                        for train in training_set_split(training_data)])]
    (results, dt) = max(trees, key=lambda xx: xx[0]['accuracy'])

    print "Chosen classifier has accuracy of %s." % results['accuracy']

    orig = results['original']
    if orig:
        os.system("mv %s classified-original.dot" % orig)
    pruned = results['pruned']
    if pruned:
        os.system("mv %s classified-pruned.dot" % pruned)
    if size(data, 0) > 0:
        classified = classify(dt, data)

        num_spam = 0
        fp = open("classified.txt", "w")
        for (row, (last_split, node)) in classified:
            comment = ""
            if True:
                comment = " # %s" % last_split
            print >> fp, "%d %d %.4f%s" % (row, node.is_spam(), node.ratio(),
                                           comment)
            if node.is_spam():
                num_spam += 1
        print "Classified as spam: %d / %d" % (num_spam, len(classified))
        res = check_results(classified, data)
        dump_results(res, give_diff=opts.diff)
        fp.close()
