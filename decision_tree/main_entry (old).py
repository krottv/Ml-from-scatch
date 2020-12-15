from decision_tree import CustomDecisionTree
import weight_height_fetcher
import decision_tree_visualizer
from sklearn.metrics import f1_score


features_train, features_valid, target_train, target_valid = weight_height_fetcher.get_splitted_data(nsamples=1000)

model = CustomDecisionTree(max_depth=10, debug=False)
model.fit(features_train, target_train)
predicted  = model.predict(features_valid)

print(f'predicted {predicted}')

score = f1_score(target_valid, predicted)
print('finish predicting values f1 = {:.3f}'.format(score))

color_classes = {'Male': 'blue', 'Female': 'red'}
decision_tree_visualizer.visualize_result(model.root_node, color_classes, save_pdf=True)
