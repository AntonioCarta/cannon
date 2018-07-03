from cannon.experiment import ModelSelectionConfig, ModelSelection


class MockData():
    def get_data(self):
        return [], []


class MockModel:
    def __init__(self, x):
        self.best_result = {
                'tr_loss': x,
                'tr_acc': x,
                'vl_loss': x,
                'vl_acc': x
            }

    def fit(self, tr, vl):
        return [self.best_result['tr_loss']], [self.best_result['vl_loss']]


class MockBuilder:
    def __init__(self):
        pass

    def build_model(self, params):
        m = MockModel(params['x'])
        return m

    def generate_params(self, i):
        return {'x': i+1}

    def total_grid_configurations(self):
        return len(self.h_iter) * len(self.m_iter)


def test_grid_search():
    exp = ModelSelection(log_dir='./logs/debug/')
    builder = MockBuilder()
    config = ModelSelectionConfig(builder, MockData(), MockData(), 0, 10, 3)

    exp.run(config)
    assert exp.best_results['vl_loss_avg'] == 0


if __name__ == '__main__':
    test_grid_search()