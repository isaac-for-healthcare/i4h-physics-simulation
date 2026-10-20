# External composition example

Install this directory separately after installing the physics, endoluminal and surgical wheels, then run `i4h-mount-example`. All rod, tet and jaw geometry is generated; no production assets are replaced or required.

At `step_start`, the custom OneWayMount copies the last completed rod-tip position into tool commands before instrument control executes. Its ordinary MountState sidecar owns grip, sample count and copied pose arrays and resets them explicitly. The example grasps, displaces and releases a block with two isolated environments in eager and captured execution. This is an explicitly one-way mount; tool reaction is not fed back into the rod. It does not demonstrate two-way robot coupling.
