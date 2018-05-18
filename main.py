from sklearn.linear_model  import LogisticRegression, Perceptron
from sklearn.svm           import SVC
from utils.dataset         import Dataset
from utils.improved        import Improved
from utils.scorer          import report_score
from numpy import cumsum


#languages = ['english', 'spanish']
languages = ['english']
#languages = ['spanish']

macroF1  = []
accuracy = []
results  = []

failures  = {}.fromkeys(languages, {})
graphData = {}.fromkeys(languages, {})
for language in languages:
    # load datasets
    data = Dataset(language)
    print('====================')
    print("{}: {} training - {} dev - {} test".format(language, len(data.trainset), len(data.devset), len(data.testset)))
    dataSets = {'dev': data.devset, 'test': data.testset}

    ###########################################################################
    if language == 'english':
        models   = [3]#[1, 2, 4, 5, 6, 7, 8]
        modelsEN = models
    if language == 'spanish':
        models   = [1, 2, 4, 5, 6, 9]
        modelsES = models
    ###########################################################################
    for dataSetName, dataSet in dataSets.items():
        for model in models:
            ###################################################################
            if   model == 1:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens'])
        
            elif model == 2:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens', 'syllables'])
            
            elif model == 3:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens', 'concreteness'])
        
            elif model == 4:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens', 'embeddings'])
        
            elif model == 5:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens', 'pos'])
        
            elif model == 6:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens', 'frequency'])
            
            elif model == 7:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens', 'syllables', 'concreteness', 'frequency'])
            
            elif model == 8:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens', 'syllables', 'concreteness', 'embeddings', 'pos', 'frequency'])

            elif model == 9:
                scheme = Improved(language, model=SVC, skipErrors=False,
                                  features=['len_chars', 'len_tokens', 'syllables', 'embeddings', 'pos', 'frequency'])

            elif model == 0:  # to skip the language
                continue
            #######################################################################
            scheme.train(data.trainset)
            predictions = scheme.test(dataSet)
            gold_labels = [sent['gold_label'] for sent in dataSet]

            # collect data for the training rate graph
            gmodel = language + str(model) + dataSetName
            graphData[gmodel] = cumsum([ pred == sent['gold_label']
                    for sent, pred in zip(dataSet, predictions) ]) / range(1,len(dataSet)+1)

            # collect and print results
            print("Using model", model)
            macroF1, accuracy = report_score(gold_labels, predictions)
            results.append([model, macroF1, 100.*accuracy, language, dataSetName])

            # log failures
            model = language + str(model) + dataSetName
            failures[model] = [
                    (sent['gold_label'], sent['target_word'])
                    for pred, sent in zip(predictions, dataSet)
                    if pred != sent['gold_label'] ]
            
            failures[model] = sorted(failures[model], key=lambda x: x[0])  # sort alphabetically

# print training rate graphs
from pylab import *
for language in languages:
    if   language == 'english':
        models = modelsEN
    elif language == 'spanish':
        models = modelsES
    for dataSetName in dataSets.keys():
        for model in models:
            gmodel = language + str(model) + dataSetName
            print(gmodel)
            glabel = language + ' model ' + str(model) + ', ' + dataSetName + ' set'
            plot(100.*graphData[gmodel][5:1000], label=glabel)  # only look at first 1000, except the first 10
            xlabel('Number of Training Instances')
            ylabel('Cumulative Accuracy [%]')
            title('Learning Rate Graph')
            ylim([50.,100.])  # the limits of the graph
        legend(loc=4)
        show()

# print results table
print('#', 'F1', 'acc', 'lang', 'data', sep='\t')
for row in results:
    print('{}\t{:5.3f}\t{:4.1f} %\t{}\t{}'.format(*row))

# function to print failures
def print_failures(language, model, dataSetName):
    model = language + str(model) + dataSetName
    previousRow = None
    for row in failures[model]:
        if row != previousRow:
            print(row)
            previousRow = row

print_failures('english', 3, 'dev')
#print_failures('english', 1, 'test')
