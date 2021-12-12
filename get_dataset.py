import minerl
from collections import deque
from copy import deepcopy

from dataset import Transition

from collections import OrderedDict


def put_data_into_dataset(env_name, action_manager, dataset, minecraft_human_data_dir,
                          continuous_action_stacking_amount=3,
                          only_successful=True, max_duration_steps=None, max_reward=256.,
                          test=False):

    print(f"\n Adding data from {env_name} \n")

    treechop_data = env_name == "MineRLTreechop-v0"

    def is_success(sample):
        if max_duration_steps is None:
            return sample[-1]['success']
        else:
            return sample[-1]['success'] and sample[-1]['duration_steps'] < max_duration_steps

    def is_no_op(sample):
        action = sample[1]
        a_id = action_manager.get_id(action)
        assert type(a_id) == int
        return a_id == 0 
    def process_sample(sample, last_reward):
        reward = sample[2]

        if reward > last_reward:
            last_reward = reward

        gatherlog_sample = last_reward < 2.

        if treechop_data:
            for key, value in action_manager.zero_action.items():
                if key not in sample[1]:
                    sample[1][key] = value

            sample[0]['equipped_items'] = OrderedDict([(
                'mainhand',
                OrderedDict([('damage', 0), ('maxDamage', 0), ('type', 0)])
            )])

            sample[0]["inventory"] = OrderedDict([
                ('coal', 0),
                ('cobblestone', 0),
                ('crafting_table', 0),
                ('dirt', 0),
                ('furnace', 0),
                ('iron_axe', 0),
                ('iron_ingot', 0),
                ('iron_ore', 0),
                ('iron_pickaxe', 0),
                ('log', 0),
                ('planks', 0),
                ('stick', 0),
                ('stone', 0),
                ('stone_axe', 0),
                ('stone_pickaxe', 0),
                ('torch', 0),
                ('wooden_axe', 0),
                ('wooden_pickaxe', 0)
            ])

        if reward != 0.:
            if reward > max_reward:
                counter_change = - dataset.remove_new_data()
            else:
                dataset.append_sample(sample, gatherlog_sample, treechop_data)
                dataset.update_last_reward_index()
                counter_change = 1
        else:
            if not is_no_op(sample) or sample[4]:  
                dataset.append_sample(sample, gatherlog_sample, treechop_data)
                counter_change = 1
            else:
                counter_change = 0

        return counter_change, last_reward

    data = minerl.data.make(env_name, data_dir=minecraft_human_data_dir)
    trajs = data.get_trajectory_names()

    sample_que = deque(maxlen=continuous_action_stacking_amount)

    total_trajs_counter = 0
    added_sample_counter = 0

    initial_sample_amount = dataset.transitions.current_size()

    for n, traj in enumerate(trajs):
        for j, sample in enumerate(data.load_data(traj, include_metadata=True)):
            if j == 0:
                print(sample[-1])

                if only_successful:
                    if not is_success(sample):
                        print("skipping trajectory")
                        break

                total_trajs_counter += 1
                last_reward = 0.

            sample_que.append(sample)

            if len(sample_que) == continuous_action_stacking_amount:
                for i in range(1, continuous_action_stacking_amount):
                    sample_que[0][1]['camera'] += sample_que[i][1]['camera']

                    if sample_que[i][2] != 0.: 
                        break

                added_samples, last_reward = process_sample(sample_que[0], last_reward)

                added_sample_counter += added_samples

        if len(sample_que) > 0: 
            for i in range(1, continuous_action_stacking_amount):
                added_samples, last_reward = process_sample(sample_que[i], last_reward)
                added_sample_counter += added_samples

            added_sample_counter -= dataset.remove_new_data()

            last_transition = deepcopy(dataset.transitions.data[dataset.transitions.index - 1])
            dataset.transitions.data[dataset.transitions.index - 1] = \
                Transition(last_transition.state, last_transition.vector,
                           last_transition.action, last_transition.reward, False)

        sample_que.clear()

        print(f"{n+1} / {len(trajs)}, added: {total_trajs_counter}")
        assert dataset.transitions.current_size() - initial_sample_amount == added_sample_counter

        if test:
            if total_trajs_counter >= 2:
                assert total_trajs_counter == 2
                break
