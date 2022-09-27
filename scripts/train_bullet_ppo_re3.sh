for env_id in Ant Walker2D
do
  python examples/ppo_re3_bullet.py --action-space cont --env-id ${env_id}BulletEnv-v0 --algo ppo --n-envs 10 --exploration re3 --total-time-steps 2000000 --n-steps 128
done