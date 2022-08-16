from numerical import Torch, Group, activators

def build_group(name, shape, parent=None):
	if parent:
		group = Group(name, Torch.set_embedding(Torch.parameterize_like(parent.embedding)))
	group = Group(name, Torch.set_embedding(Torch.random_tensor(shape, 'float')))
	group.add_attribute(
		Torch.set_functional(Group.from_preset('relu')),
		Torch.set_functional(Group.from_preset('linear')),
		Torch.set_functional(Group.from_preset('relu')),
		Torch.set_functional(Group.from_preset('linear'))
	)
	group.auto_compose(out_projection_type='logits', resizing=True)
	group.compile()
	return group

def build_model(input_shape, output_shape):
	model = Group('model')
	model.add_attribute(
		build_group('group_1', (input_shape, 32)),
		build_group('group_2', (32, output_shape))
	)
	model.auto_compose(out_projection_type='logits', resizing=True)
	model.compile()
	return model

model = build_model(input_shape, output_shape)
