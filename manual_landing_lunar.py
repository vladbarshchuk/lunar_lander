import gymnasium as gym
import pygame
import numpy as np

# Initialize Pygame and its font module
pygame.init()
pygame.font.init()

# Set up the Pygame window.
screen_width, screen_height = 600, 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Lunar Lander Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

def display_landing_info(episode_reward, fuel_consumed):
    """Display the landing score and fuel consumption in the terminal."""
    print(f"Landing Score: {episode_reward}, Fuel Consumed: {fuel_consumed}")

def run_game():
    """Run one episode of the LunarLander game and return the episode reward."""
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, info = env.reset()
    episode_reward = 0
    done = False
    fuel_consumed = 0  # Track fuel consumption based on thruster usage

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_UP]:
            action = 2
        elif keys[pygame.K_RIGHT]:
            action = 3
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

        # Estimate fuel consumption: count thruster actions (actions 1, 2, and 3)
        if action in [1, 2, 3]:
            fuel_consumed += 1

        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(frame)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    display_landing_info(episode_reward, fuel_consumed)
    env.close()
    return episode_reward

def show_end_screen(episode_reward):
    if episode_reward >= 200:
        message = "Congratulations! Successful landing!"
    else:
        message = "Landing was not successful. Try again!"

    screen.fill((0, 0, 0))
    message_surface = font.render(message, True, (255, 255, 255))
    screen.blit(message_surface, (20, 20))

    button_width, button_height = 100, 50
    button_rect = pygame.Rect(
        (screen_width - button_width) // 2,
        (screen_height - button_height) // 2,
        button_width,
        button_height,
    )
    pygame.draw.rect(screen, (0, 255, 0), button_rect)

    button_text = font.render("Replay", True, (0, 0, 0))
    text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, text_rect)

    pygame.display.flip()
    return button_rect

def wait_for_replay(button_rect):
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    return True
        clock.tick(30)
    return False

def main():
    replay = True
    while replay:
        episode_reward = run_game()
        button_rect = show_end_screen(episode_reward)
        replay = wait_for_replay(button_rect)

    pygame.quit()

if __name__ == "__main__":
    main()

