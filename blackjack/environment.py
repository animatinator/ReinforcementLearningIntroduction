# Blackjack environment
from numpy import random

# 1-9 and four tens
BLACKJACK_CARDS = list(range(1, 10)) + [10]*4

class TimeStep:
	def __init__(self, observation, reward, terminal):
		self.observation = observation
		self.reward = reward
		self.terminal = terminal
	
	def __str__(self):
		return "{}: {}, reward: {}".format("Terminal state" if self.terminal else "State", self.observation, self.reward)

class State:
	def __init__(self, initial_hand):
		self._hand = initial_hand
		
	def deal(self, card):
		self._hand.append(card)
		
	def usable_ace(self):
		return 1 in self._hand and sum(self._hand) + 10 <= 21
		
	def sum(self):
		if self.usable_ace():
			return sum(self._hand) + 10
		return sum(self._hand)
		
	def last_card(self):
		return self._hand[-1]

	def bust(self):
		return self.sum() > 21

class Blackjack:
	def __init__(self):
		self._reset()
	
	def _make_observation(self):
		return (self._player.sum(), self._dealer.last_card(), self._player.usable_ace())
		
	def _reset(self):
		player_cards = self._draw_hand()
		dealer_cards = self._draw_hand()
		
		while sum(player_cards) < 12:
			player_cards.append(self._draw())
		
		self._player = State(player_cards)
		self._dealer = State(dealer_cards)
		
	def reset(self):
		self._reset()
		
		return TimeStep(self._make_observation(), 0, False)
		
	def _draw_hand(self):
		return [self._draw(), self._draw()]
	
	def _draw(self):
		return random.choice(BLACKJACK_CARDS)
		
	def _compute_score(self, player_score, dealer_score):
		if player_score == dealer_score:
			return 0
		elif player_score > dealer_score:
			return 1
		else:
			return -1
		
	def step(self, stick):
		done = False

		if stick:
			self._player.deal(self._draw())
			
			if self._player.bust():
				reward = -1
				done = True
			else:
				reward = 0
		else:
			done = True
			while self._dealer.sum() < 17:
				self._dealer.deal(self._draw())
			reward = self._compute_score(self._player.sum(), self._dealer.sum())
		
		return TimeStep(self._make_observation(), reward, done)
