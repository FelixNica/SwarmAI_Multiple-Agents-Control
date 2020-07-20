import pygame
from numpy import random


class Bot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.full = False


class Pack:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.loaded_in = 0
        self.unloaded_in = 0


class Place:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Game:
    def __init__(self,
                 bots_number,
                 packs_number,
                 places_number,
                 bot_scale=0.03,
                 pack_scale=0.025,
                 place_scale=0.1,
                 load_reward=50,
                 unload_reward=100,
                 episode_steps=50,
                 render_scale=6):

        pygame.init()

        self.space_size = 100
        self.bot_size = self.space_size * bot_scale
        self.pack_size = self.space_size * pack_scale
        self.place_size = self.space_size * place_scale
        self.load_reward = load_reward
        self.unload_reward = unload_reward
        self.render_scale = render_scale
        self.screen = pygame.display.set_mode((self.space_size*self.render_scale,
                                               self.space_size*self.render_scale))

        self.episode_steps = episode_steps
        self.step_counter = 0

        self.move_coefficient = 0.1
        self.bot_color = (52, 171, 235)
        self.pack_color = (235, 150, 52)
        self.place_color = (155, 235, 52)
        self.n_bots = bots_number
        self.n_packs = packs_number
        self.n_places = places_number
        self.bots = []
        self.packs = []
        self.places = []
        self.heading = []
        init_state = self.reset()
        self.state_shape = len(init_state)
        self.action_shape = len(self.heading)

        self.max_reward = self.n_packs*(self.load_reward + self.unload_reward)
        self.screen_index = 0

    @staticmethod
    def seed(seed):
        random.seed(seed)

    def reset(self):
        self.step_counter = 0
        border = int(self.space_size / 10)
        max_pos = int(self.space_size - border)
        bots_pos = random.randint(border, max_pos, (self.n_bots, 2))
        packs_pos = random.randint(border, max_pos, (self.n_packs, 2))
        places_pos = random.randint(border, max_pos, (self.n_places, 2))

        self.bots = []
        for o in range(self.n_bots):
            obj = Bot(bots_pos[o][0], bots_pos[o][1])
            self.bots.append(obj)
        self.packs = []
        for o in range(self.n_packs):
            obj = Pack(packs_pos[o][0], packs_pos[o][1])
            self.packs.append(obj)
        self.places = []
        for o in range(self.n_places):
            obj = Place(places_pos[o][0], places_pos[o][1])
            self.places.append(obj)
        self.heading = []
        for o in bots_pos:
            self.heading.append(o[0])
            self.heading.append(o[1])

        return self.make_state()

    def generate_action(self):
        bots_pos = random.randint(0, self.space_size, (self.n_bots, 2))
        test_action = []
        for i in range(self.n_bots):
            test_action.append(bots_pos[i][0])
            test_action.append(bots_pos[i][1])
        return test_action

    def make_state(self):
        state = []
        for bot in self.bots:
            state.append(bot.x)
            state.append(bot.y)
            if bot.full:
                state.append(1)
            else:
                state.append(0)
        for pack in self.packs:
            state.append(pack.x)
            state.append(pack.y)
            state.append(pack.loaded_in)
            state.append(pack.unloaded_in)
        for place in self.places:
            state.append(place.x)
            state.append(place.y)
        for value in self.heading:
            state.append(value)

        return state

    def step(self, action):
        for event in pygame.event.get():
            pass

        self.heading = action
        # Move to heading
        for i, bot in enumerate(self.bots):

            delta_x = self.heading[i*2] - bot.x
            delta_y = self.heading[i*2+1] - bot.y

            bot.x, bot.y = \
                int(bot.x + self.move_coefficient * delta_x), \
                int(bot.y + self.move_coefficient * delta_y)

        # Process Packs
        reward = 0
        done = False

        unloaded = 0
        for i, pack in enumerate(self.packs):
            for b, bot in enumerate(self.bots):
                if not bot.full \
                        and(pack.x-bot.x)**2 <= self.bot_size**2 \
                        and(pack.y-bot.y)**2 <= self.bot_size**2 \
                        and pack.loaded_in is 0 \
                        and pack.unloaded_in is 0:
                        pack.loaded_in = b + 1
                        bot.full = True
                        reward += self.load_reward
                        #print(f"Loaded Pack {i} in Bot {b}")

            if pack.loaded_in is not 0:
                pack.x = self.bots[pack.loaded_in-1].x
                pack.y = self.bots[pack.loaded_in-1].y

            for p, place in enumerate(self.places):
                if pack.loaded_in is not 0 \
                        and (pack.x-place.x)**2 <= ((self.place_size-self.pack_size)/2)**2 \
                        and (pack.y-place.y)**2 <= ((self.place_size-self.pack_size)/2)**2:
                    pack.unloaded_in = p + 1
                    self.bots[pack.loaded_in-1].full = False
                    pack.loaded_in = 0
                    reward += self.unload_reward
                    #print(f"Unloaded Pack {i} in Place {p}")

            if pack.unloaded_in is not 0:
                unloaded += 1

        if unloaded == self.n_packs:
            done = True
            print(f"Done: All Packs are Unloaded: {unloaded}")
        self.step_counter += 1
        if self.step_counter >= self.episode_steps:
            done = True
            print(f"Done: Max steps reached, Unloaded: {unloaded}")

        return self.make_state(), reward, done

    def render(self, save_location=None):
        rs = self.render_scale
        self.screen.fill((0, 0, 0))

        for bot in self.bots:
            pygame.draw.circle(self.screen,
                               (250, 250, 250),
                               (bot.x*rs, bot.y*rs),
                               int((self.bot_size*1.2)*rs))
            pygame.draw.circle(self.screen,
                               (0, 0, 0),
                               (bot.x*rs, bot.y*rs),
                               int(self.bot_size*rs))
            pygame.draw.circle(self.screen,
                               self.bot_color,
                               (bot.x*rs, bot.y*rs),
                               int(self.bot_size*0.92*rs))

        for place in self.places:
            pygame.draw.rect(self.screen,
                             self.place_color,
                             pygame.Rect((place.x - self.place_size/2)*rs,
                                         (place.y - self.place_size/2)*rs,
                                         self.place_size*rs,
                                         self.place_size*rs),
                             int(self.place_size*0.05*rs))

        for pack in self.packs:
            pygame.draw.rect(self.screen,
                             (0, 0, 0),
                             pygame.Rect((pack.x - (self.pack_size*1.2)/2)*rs,
                                         (pack.y - (self.pack_size*1.2)/2)*rs,
                                         self.pack_size*1.2*rs,
                                         self.pack_size*1.2*rs))
            pygame.draw.rect(self.screen,
                             self.pack_color,
                             pygame.Rect((pack.x - self.pack_size/2)*rs,
                                         (pack.y - self.pack_size/2)*rs,
                                         self.pack_size*rs,
                                         self.pack_size*rs))
            pygame.draw.circle(self.screen,
                               (0, 0, 0),
                               (pack.x*rs, pack.y*rs),
                               int(self.pack_size*0.2*rs))
        pygame.display.flip()
        if save_location is not None:
            pygame.image.save(self.screen, f'{save_location}{self.screen_index}.png')
            self.screen_index += 1

'''
game = Game(3, 6, 3)
change_every = 50
game.reset()
action = game.generate_action()

for i in range(100000):
    if i % change_every == 0:
        action = game.generate_action()
    state, reward, done = game.step(action)
    #print(state)
    #print('stateln', len(state))
    #print('stateshp', game.state_shape)
    #if reward > 0:
    #    print(reward)

    game.render()
    if done:
        break
    time.sleep(0.1)


#test code
pressed = pygame.key.get_pressed()
if pressed[pygame.K_UP]: self.bots[0].y -= 3
if pressed[pygame.K_DOWN]: self.bots[0].y += 3
if pressed[pygame.K_LEFT]: self.bots[0].x -= 3
if pressed[pygame.K_RIGHT]: self.bots[0].x += 3
'''
