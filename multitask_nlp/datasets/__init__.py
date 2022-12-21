from multitask_nlp.datasets.multitask_dataset import ProportionalSamplingMTDataset, \
    RoundRobinMTDataset, SamplingMTDataset

multitask_datasets = {
    'round_robin': RoundRobinMTDataset,
    'sampling': SamplingMTDataset,
    'proportional_sampling': ProportionalSamplingMTDataset,
    'annealing_sampling': ProportionalSamplingMTDataset,
    'dynamic_temperature_sampling': ProportionalSamplingMTDataset
}
