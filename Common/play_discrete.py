# from mujoco_py.generated import const
#from mujoco_py import GlfwContext
import cv2
import numpy as np
import os

#GlfwContext(offscreen=True)


class Play_Discrete:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Vid/"):
            os.mkdir("Vid/")

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):

        for z in range(self.n_skills):
            video_writer = cv2.VideoWriter(f"Vid/skill{z}" + ".avi", self.fourcc, 50.0, (250, 250))
            s, _ = self.env.reset()
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            done = False
            i = 0
            while not done:
                action = round(self.agent.choose_action(s)[0])
                
                print(action)
                #action = np.argmax(action)
                s_, r, done, _, _ = self.env.step(action)
                
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                if done or i == 200:
                    break
                s = s_
                #I = self.env.render(mode='rgb_array')
                #I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                #I = cv2.resize(I, (250, 250))
                #video_writer.write(I)
                i+=1
                
            print(f"skill: {z}, action: {action},episode reward:{episode_reward:.1f}")
            #video_writer.release()
        self.env.close()
        #cv2.destroyAllWindows()
