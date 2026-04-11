"""
FastAPI application for the KubeCost Environment.
"""

from openenv.core.env_server.http_server import create_app

try:
    from k8s_cost_optimizer.models import Action, Observation
    from k8s_cost_optimizer.server.k8s_cost_optimizer_environment import (
        K8sCostOptimizerEnvironment,
    )
except ImportError:
    from models import Action, Observation
    from server.k8s_cost_optimizer_environment import K8sCostOptimizerEnvironment

app = create_app(
    K8sCostOptimizerEnvironment,
    Action,
    Observation,
    env_name="k8s_cost_optimizer",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)
