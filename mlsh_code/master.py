import gym
# import test_envs
import tensorflow as tf
import rollouts
from policy_network import Policy
from subpolicy_network import SubPolicy
from observation_network import Features
from learner import Learner
import rl_algs.common.tf_util as U
import numpy as np
# from tinkerbell import logger
import pickle
import statistics


def start(callback, args, workerseed, rank, comm):
    env = gym.make(args.task)
    env.seed(workerseed)
    np.random.seed(workerseed)
    ob_space = env.observation_space
    ac_space = env.action_space

    num_subs = args.num_subs
    macro_duration = args.macro_duration
    num_rollouts = args.num_rollouts
    warmup_time = args.warmup_time
    train_time = args.train_time
    sub_hidden_sizes = args.sub_hidden_sizes
    sub_policy_costs = args.sub_policy_costs

    num_batches = 15

    # observation in.
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
    # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, 104])

    # features = Features(name="features", ob=ob)
    policy = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)
    old_policy = Policy(name="old_policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)

    sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=sub_hidden_sizes[x], num_hid_layers=2) for x in range(num_subs)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=sub_hidden_sizes[x], num_hid_layers=2) for x in range(num_subs)]

    learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64, args=args)
    rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts,
                                              stochastic=True, args=args, sub_policy_costs=sub_policy_costs)
    fixed_policy_rollouts = []
    for i in range(num_subs):
      fixed_policy_rollouts.append(rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts,
                                                                   stochastic=True, args=args, sub_policy_costs=sub_policy_costs, fixed_policy=i))



    for x in range(1):
        callback(x)
        if x == 0:
            learner.syncSubpolicies()
            print("synced subpols")
        # Run the inner meta-episode.

        policy.reset()
        learner.syncMasterPolicies()

        try:
            env.env.randomizeCorrect()
            shared_goal = comm.bcast(env.env.realgoal, root=0)
            env.env.realgoal = shared_goal
        except:
            pass

        # print("It is iteration %d so i'm changing the goal to %s" % (x, env.env.realgoal))
        mini_ep = 0 if x > 0 else -1 * (rank % 10)*int(warmup_time+train_time / 10)
        # mini_ep = 0

        totalmeans = []
        while mini_ep < warmup_time+train_time:
            mini_ep += 1
            # rollout
            rolls = rollout.__next__()
            allrolls = []
            allrolls.append(rolls)
            # train theta
            rollouts.add_advantage_macro(rolls, macro_duration, 0.99, 0.98)
            gmean, lmean = learner.updateMasterPolicy(rolls)
            # train phi
            test_seg = rollouts.prepare_allrolls(allrolls, macro_duration, 0.99, 0.98, num_subpolicies=num_subs)
            learner.updateSubPolicies(test_seg, num_batches, (mini_ep >= warmup_time))
            # learner.updateSubPolicies(test_seg,
            # log
            # print(("%d: global: %s, local: %s" % (mini_ep, gmean, lmean)))
            print(("Episode %d return: %s" % (mini_ep, lmean)))
            if args.s:
                totalmeans.append(gmean)
                with open('outfile'+str(x)+'.pickle', 'wb') as fp:
                    pickle.dump(totalmeans, fp)

            # evaluate sub-policies seperately
            if mini_ep % 50 == 0:
                if args.num_subs != 1:
                    print("macro acts:", rolls['macro_ac'])
                for i, fix_policy_rollout in enumerate(fixed_policy_rollouts):
                    collected_rolls = []
                    for _ in range(10):
                        collected_rolls.extend(fix_policy_rollout.__next__()['ep_rets_without_cost'])
                    print("sub %d: %.3f" % (i, statistics.mean(collected_rolls)), end=', ')
                print()


