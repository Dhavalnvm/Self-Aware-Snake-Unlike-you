import pygame
import random
import numpy as np
import sys
import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
GRID_WIDTH, GRID_HEIGHT = 20, 20
CELL_SIZE = 20
WIDTH, HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
OBSTACLE_COUNT = 10
FPS_TRAINING = 60
FPS_PLAYING = 10
SNAKE_START_RANGE = (5, 15)  # Range for random start position

# Pygame init
pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Snake")
FONT = pygame.font.SysFont("Arial", 20)

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)  # For game over message

# RL Params
ACTIONS = ['straight', 'left', 'right']
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
DIRECTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

# Using defaultdict to avoid explicit initialization
q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
RANDOM_ACTION_CHANCE = 0.05  # For AI play mode

# State encoding is more sophisticated
# We'll track 8 directions for obstacles and food distance
VISION_DISTANCE = 5  # How far the snake can "see"


def draw(snake, food, score, obstacles, game_over=False, message=""):
    """Draw the game state to the screen"""
    SCREEN.fill(BLACK)

    # Draw obstacles
    for obs in obstacles:
        pygame.draw.rect(SCREEN, GRAY, (obs[0] * CELL_SIZE, obs[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw snake
    for i, (x, y) in enumerate(snake):
        color = BLUE if i == 0 else GREEN
        pygame.draw.rect(SCREEN, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw food
    pygame.draw.rect(SCREEN, RED, (food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw score
    text = FONT.render(f"Score: {score}", True, WHITE)
    SCREEN.blit(text, (10, 10))

    # Draw game over message if applicable
    if game_over:
        text = FONT.render(message, True, YELLOW)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        SCREEN.blit(text, text_rect)

        # Instructions for restart
        restart_text = FONT.render("Press R to restart or Q to quit", True, WHITE)
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30))
        SCREEN.blit(restart_text, restart_rect)

    pygame.display.flip()


def turn(direction_idx, action):
    """Apply an action to change direction"""
    if action == 'left': return (direction_idx - 1) % 4
    if action == 'right': return (direction_idx + 1) % 4
    return direction_idx


def move(pos, direction_idx):
    """Move in the current direction"""
    dx, dy = DIRECTIONS[direction_idx]
    return (pos[0] + dx, pos[1] + dy)


def is_collision(pos, snake, obstacles):
    """Check if position collides with snake, obstacles or walls"""
    # Wall collision
    if not (0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT):
        return True
    # Snake collision (except tail which will move)
    if pos in snake[:-1]:
        return True
    # Obstacle collision
    if pos in obstacles:
        return True
    return False


def get_state(snake, food, direction_idx, obstacles):
    """Create a more detailed state representation"""
    head = snake[0]

    # Check for immediate dangers (adjacent cells)
    straight_pos = move(head, direction_idx)
    left_dir = turn(direction_idx, 'left')
    right_dir = turn(direction_idx, 'right')
    left_pos = move(head, left_dir)
    right_pos = move(head, right_dir)

    danger_straight = is_collision(straight_pos, snake, obstacles)
    danger_left = is_collision(left_pos, snake, obstacles)
    danger_right = is_collision(right_pos, snake, obstacles)

    # Determine food direction
    # This creates 8 possible food directions (N, NE, E, SE, S, SW, W, NW)
    rel_x = food[0] - head[0]
    rel_y = food[1] - head[1]

    # Normalize to -1, 0, 1 for each direction
    if rel_x > 0:
        rel_x = 1
    elif rel_x < 0:
        rel_x = -1

    if rel_y > 0:
        rel_y = 1
    elif rel_y < 0:
        rel_y = -1

    # Check food distance (limited to VISION_DISTANCE)
    food_dist_x = min(abs(food[0] - head[0]), VISION_DISTANCE)
    food_dist_y = min(abs(food[1] - head[1]), VISION_DISTANCE)

    # Current direction
    dir_up = 1 if direction_idx == 0 else 0
    dir_right = 1 if direction_idx == 1 else 0
    dir_down = 1 if direction_idx == 2 else 0
    dir_left = 1 if direction_idx == 3 else 0

    # Combine all state components
    state = (
        danger_straight, danger_left, danger_right,
        rel_x, rel_y,
        dir_up, dir_right, dir_down, dir_left
    )

    return state


def place_food(snake, obstacles):
    """Place food in an empty cell"""
    empty_cells = []
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            pos = (x, y)
            if pos not in snake and pos not in obstacles:
                empty_cells.append(pos)

    if not empty_cells:  # If no empty cells (very rare)
        return None

    return random.choice(empty_cells)


def place_obstacles(num=OBSTACLE_COUNT):
    """Place obstacles randomly but not too close to center"""
    obstacles = []
    center = (GRID_WIDTH // 2, GRID_HEIGHT // 2)

    for _ in range(num):
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            distance_to_center = abs(pos[0] - center[0]) + abs(pos[1] - center[1])

            # Ensure obstacles aren't too close to center
            if distance_to_center > 3 and pos not in obstacles:
                obstacles.append(pos)
                break

    return obstacles


def save_qtable():
    """Save Q-table to file"""
    # Convert defaultdict to regular dict for saving
    dict_qtable = dict(q_table)
    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict_qtable, f)
    print(f"Q-table saved with {len(dict_qtable)} states")


def load_qtable():
    """Load Q-table from file"""
    global q_table
    if os.path.exists("q_table.pkl"):
        with open("q_table.pkl", "rb") as f:
            loaded_dict = pickle.load(f)
            # Convert back to defaultdict
            q_table = defaultdict(lambda: np.zeros(len(ACTIONS)), loaded_dict)
        print(f"Q-table loaded with {len(q_table)} states")
    else:
        print("No Q-table found, starting fresh")


def get_reward(new_head, snake, food, collision):
    """Calculate reward for current action"""
    if collision:
        return -10  # Penalty for collisions

    if new_head == food:
        return 10  # Reward for eating food

    # Small penalty for each move to encourage efficient paths
    # Calculate if we're getting closer to food
    head = snake[0]
    prev_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
    new_dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])

    if new_dist < prev_dist:
        return 0.1  # Small reward for moving closer to food
    else:
        return -0.1  # Small penalty for moving away from food


def train_snake(episodes=MAX_EPISODES, visualize=False):
    """Train the snake using Q-learning"""
    global q_table
    epsilon = INITIAL_EPSILON
    scores = []
    max_score = 0
    clock = pygame.time.Clock()

    for episode in range(1, episodes + 1):
        # Initialize snake with random start position
        start_x = random.randint(SNAKE_START_RANGE[0], SNAKE_START_RANGE[1])
        start_y = random.randint(SNAKE_START_RANGE[0], SNAKE_START_RANGE[1])
        snake = [(start_x, start_y), (start_x - 1, start_y)]
        direction_idx = 1  # Start moving right

        obstacles = place_obstacles()
        food = place_food(snake, obstacles)
        score = 0
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            pygame.event.pump()  # Keep pygame responding

            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    save_qtable()
                    pygame.quit()
                    sys.exit()

            # Get current state
            state = get_state(snake, food, direction_idx, obstacles)

            # Choose action (epsilon-greedy policy)
            if random.random() < epsilon:
                action_idx = random.randint(0, len(ACTIONS) - 1)  # Explore
            else:
                action_idx = np.argmax(q_table[state])  # Exploit

            # Apply action
            new_direction_idx = turn(direction_idx, ACTIONS[action_idx])
            new_head = move(snake[0], new_direction_idx)

            # Check for collision
            collision = is_collision(new_head, snake, obstacles)

            # Calculate reward
            reward = get_reward(new_head, snake, food, collision)
            total_reward += reward

            # Update snake position if no collision
            if not collision:
                snake.insert(0, new_head)

                # Check if food eaten
                if new_head == food:
                    score += 1
                    if score > max_score:
                        max_score = score
                    food = place_food(snake, obstacles)
                else:
                    snake.pop()  # Remove tail if no food eaten

                # Get new state
                new_state = get_state(snake, food, new_direction_idx, obstacles)

                # Update Q-table (Q-learning formula)
                q_table[state][action_idx] += LEARNING_RATE * (
                        reward + DISCOUNT_FACTOR * np.max(q_table[new_state]) - q_table[state][action_idx]
                )

                # Update direction
                direction_idx = new_direction_idx

                # Visualize if requested
                if visualize and step % 2 == 0:  # Only render every other step to speed up training
                    draw(snake, food, score, obstacles)
                    clock.tick(FPS_TRAINING)
            else:
                # Update Q-table for terminal state
                q_table[state][action_idx] += LEARNING_RATE * (reward - q_table[state][action_idx])
                break

        # Decay epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        scores.append(score)

        # Progress updates
        if episode % 50 == 0:
            avg_score = sum(scores[-50:]) / min(50, len(scores[-50:]))
            print(f"Episode {episode:4d} | Score: {score:2d} | Avg Score: {avg_score:.2f} | "
                  f"Max Score: {max_score:2d} | Epsilon: {epsilon:.4f} | "
                  f"Q-table size: {len(q_table)}")

    # Save Q-table and plot training progress
    save_qtable()

    # Plot training progress
    plt.figure(figsize=(12, 6))

    # Plot episode scores
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Training Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    # Plot moving average
    plt.subplot(1, 2, 2)
    moving_avg = []
    window_size = 100
    for i in range(len(scores) - window_size + 1):
        moving_avg.append(sum(scores[i:i + window_size]) / window_size)
    plt.plot(moving_avg)
    plt.title(f"Moving Average (Window Size: {window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")

    plt.tight_layout()
    plt.savefig("training_scores.png")
    plt.close()


def play_snake():
    """Play the game using the trained AI"""
    load_qtable()

    # Initialize snake with random start position
    start_x = random.randint(SNAKE_START_RANGE[0], SNAKE_START_RANGE[1])
    start_y = random.randint(SNAKE_START_RANGE[0], SNAKE_START_RANGE[1])
    snake = [(start_x, start_y), (start_x - 1, start_y)]
    direction_idx = 1  # Start moving right

    obstacles = place_obstacles()
    food = place_food(snake, obstacles)
    clock = pygame.time.Clock()
    score = 0
    steps = 0
    game_over = False

    while True:
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    return play_snake()  # Restart game
                elif event.key == pygame.K_q:
                    pygame.quit()
                    return

        if not game_over:
            # Get current state
            state = get_state(snake, food, direction_idx, obstacles)

            # Choose action with a small chance of random action
            if random.random() < RANDOM_ACTION_CHANCE:
                action_idx = random.randint(0, len(ACTIONS) - 1)
            else:
                action_idx = np.argmax(q_table[state])

            # Apply action
            direction_idx = turn(direction_idx, ACTIONS[action_idx])
            new_head = move(snake[0], direction_idx)

            # Check for collision
            if is_collision(new_head, snake, obstacles):
                game_over = True
                message = f"Game Over! Final Score: {score} | Steps: {steps}"
                draw(snake, food, score, obstacles, game_over, message)
                continue

            # Update snake position
            snake.insert(0, new_head)

            # Check if food eaten
            if new_head == food:
                score += 1
                food = place_food(snake, obstacles)
            else:
                snake.pop()  # Remove tail if no food eaten

            steps += 1

        # Draw game state
        draw(snake, food, score, obstacles, game_over)
        clock.tick(FPS_PLAYING)


def human_play():
    """Let a human play the game"""
    # Initialize snake in the middle
    middle_x, middle_y = GRID_WIDTH // 2, GRID_HEIGHT // 2
    snake = [(middle_x, middle_y), (middle_x - 1, middle_y)]
    direction_idx = 1  # Start moving right

    obstacles = []  # No obstacles in human mode for simplicity
    food = place_food(snake, obstacles)
    clock = pygame.time.Clock()
    score = 0
    steps = 0
    game_over = False

    while True:
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if not game_over:
                    # Change direction based on key press
                    if event.key == pygame.K_w and direction_idx != 2:  # Up
                        direction_idx = 0
                    elif event.key == pygame.K_d and direction_idx != 3:  # Right
                        direction_idx = 1
                    elif event.key == pygame.K_s and direction_idx != 0:  # Down
                        direction_idx = 2
                    elif event.key == pygame.K_a and direction_idx != 1:  # Left
                        direction_idx = 3

                # Restart or quit when game over
                if game_over:
                    if event.key == pygame.K_r:
                        return human_play()  # Restart game
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        return

        if not game_over:
            # Move snake
            new_head = move(snake[0], direction_idx)

            # Check for collision
            if is_collision(new_head, snake, obstacles):
                game_over = True
                message = f"Game Over! Final Score: {score} | Steps: {steps}"
                draw(snake, food, score, obstacles, game_over, message)
                continue

            # Update snake position
            snake.insert(0, new_head)

            # Check if food eaten
            if new_head == food:
                score += 1
                food = place_food(snake, obstacles)
            else:
                snake.pop()  # Remove tail if no food eaten

            steps += 1

        # Draw game state
        draw(snake, food, score, obstacles, game_over)
        clock.tick(FPS_PLAYING)


def display_menu():
    """Display a simple menu"""
    SCREEN.fill(BLACK)
    title = FONT.render("RL Snake Game", True, WHITE)
    title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 4))
    SCREEN.blit(title, title_rect)

    options = [
        ("Press T to Train AI", GREEN),
        ("Press P to Watch AI Play", BLUE),
        ("Press H to Play Yourself", RED),
        ("Press Q to Quit", WHITE)
    ]

    for i, (text, color) in enumerate(options):
        option_text = FONT.render(text, True, color)
        option_rect = option_text.get_rect(center=(WIDTH // 2, HEIGHT // 3 + i * 40))
        SCREEN.blit(option_text, option_rect)

    pygame.display.flip()


def main():
    """Main program entry point"""
    running = True
    while running:
        display_menu()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    train_snake(episodes=MAX_EPISODES, visualize=True)
                elif event.key == pygame.K_p:
                    play_snake()
                elif event.key == pygame.K_h:
                    human_play()
                elif event.key == pygame.K_q:
                    running = False

        pygame.time.wait(100)  # Prevent CPU hogging in menu

    pygame.quit()


# === Main ===
if __name__ == '__main__':
    main()
