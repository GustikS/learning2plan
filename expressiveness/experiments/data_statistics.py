from os import listdir

from learning2plan.parsing import get_datasets

folder = "../datasets/rosta"

domains = sorted(listdir(folder))
for domain in domains:
    datasets = get_datasets(folder + "/" + domain)
    avg_distances = []
    num_states = []
    num_objects = []
    for instance in datasets:
        labels = [state.label for state in instance.states]
        avg_distances.append(sum(labels) / len(labels))
        num_states.append(len(instance.states))
        num_objects.append(len(instance.domain.objects))
    print(domain, sum(num_states) / len(num_states), sum(num_objects) / len(num_objects), sum(avg_distances) / len(avg_distances))
