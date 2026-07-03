using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

public class PlayerMovement : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed;
    public float walkSpeed;
    public float sprintSpeed;
    public float groundDrag;

    [Header("Jumping")]
    public float jumpForce;
    public float jumpCooldown = 0.25f;
    public float airMultiplier;
    private float jumpCooldownTimer;

    [Header("Crouching")]
    public float crouchSpeed;
    public float crouchYScale;
    private float startYScale;

    [Header("Keybinds")]
    public KeyCode jumpKey = KeyCode.Space;
    public KeyCode sprintKey = KeyCode.LeftShift;
    public KeyCode crouchKey = KeyCode.LeftControl;

    [Header("Ground Check")]
    public float playerHeight;
    public LayerMask whatIsGround;
    bool grounded;

    [Header("Slope Handling")]
    public float maxSlopeAngle;
    private RaycastHit slopeHit;
    private bool exitingSlope;

    public Transform orientation;

    float horizontalInput;
    float verticalInput;

    Vector3 moveDir;

    Rigidbody rb;

    public MovementState state;
    public enum MovementState
    {
        walking,
        sprinting,
        crouching,
        air
    }

    [Header("Debugging")]
    [Tooltip("Enable movement/input debug prints and small on-screen overlay.")]
    public bool debugMode = false;
    [Tooltip("Seconds between debug logs when debugMode is true.")]
    public float debugLogInterval = 0.5f;
    float debugTimer = 0f;
    private int jumpAttemptCounter = 0;
    private int jumpSuccessCounter = 0;

    // cached BehaviorParameters if present so we can allow Heuristic mode even when communicator exists
    private BehaviorParameters behaviorParameters;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.freezeRotation = true;
        // Use interpolation for smoother visual motion (reduces jitter)
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        // Use Continuous collision detection to avoid tunneling at high speeds
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;

        jumpCooldownTimer = 0f;
        startYScale = transform.localScale.y;
        behaviorParameters = GetComponent<BehaviorParameters>();
    }

    void Update()
    {
        // Ground Check (keep in Update so UI/Input logic sees current ground state)
        grounded = Physics.Raycast(transform.position, Vector3.down, playerHeight * 0.5f + 0.2f, whatIsGround);

        // Decrement jump cooldown timer
        jumpCooldownTimer -= Time.deltaTime;

        // If communicator is on AND agent is not in heuristic mode -> do not read human input.
        bool isCommunicatorOn = Academy.Instance.IsCommunicatorOn;
        bool isHeuristicMode = (behaviorParameters != null &&
            behaviorParameters.BehaviorType == BehaviorType.HeuristicOnly);
        bool isInferenceOnly = (behaviorParameters != null &&
            behaviorParameters.BehaviorType == BehaviorType.InferenceOnly);

        // Nonaktifkan mouse kalau: communicator aktif dan bukan heuristic,
        // ATAU kalau inference only tanpa communicator
        if ((isCommunicatorOn && !isHeuristicMode) || isInferenceOnly)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
            return;
        }

        //if (isCommunicatorOn && !isHeuristicMode)
        //{
        //    // Training connection present and agent in inference/RL mode: ignore human input.
        //    horizontalInput = 0;
        //    verticalInput = 0;
        //    if (debugMode && debugTimer <= 0f)
        //    {
        //        Debug.Log("[PlayerMovement] Communicator ON and not heuristic -> ignoring human input.");
        //    }
        //}
        else
        {
            // Mode Normal / Heuristic: Read human input
            MyInput();
            // SpeedControl must run with physics (FixedUpdate) — don't call it from Update.
            StateHandler();
        }

        // Handle Drag
        rb.linearDamping = grounded ? groundDrag : 0f;

        // Periodic debug logging
        if (debugMode)
        {
            debugTimer -= Time.deltaTime;
            if (debugTimer <= 0f)
            {
                debugTimer = debugLogInterval;
                Debug.Log($"[PlayerMovement] Inputs H:{horizontalInput:F2} V:{verticalInput:F2} grounded:{grounded} jumpCooldown:{jumpCooldownTimer:F3} state:{state} vel:{rb.linearVelocity.magnitude:F2} communicatorOn:{isCommunicatorOn} heuristic:{isHeuristicMode}");
            }
        }
    }

    private void FixedUpdate()
    {
        // If communicator is on AND agent is not heuristic mode, still don't move under human input.
        bool isCommunicatorOn = Academy.Instance.IsCommunicatorOn;
        bool isHeuristicMode = (behaviorParameters != null && behaviorParameters.BehaviorType == BehaviorType.HeuristicOnly);

        if (!isCommunicatorOn || isHeuristicMode)
        {
            MovePlayer();
            // Run speed control in FixedUpdate so velocity manipulations are consistent with physics steps.
            SpeedControl();
        }

        if (debugMode)
        {
            // quick checks for physics anomalies
            if (float.IsNaN(rb.linearVelocity.x) || float.IsNaN(rb.linearVelocity.y) || float.IsNaN(rb.linearVelocity.z))
                Debug.LogError("[PlayerMovement] rb.linearVelocity contains NaN");

            if (rb.mass <= 0)
                Debug.LogWarning("[PlayerMovement] Rigidbody mass <= 0 (unexpected)");
        }
    }

    void MyInput()
    {
        horizontalInput = Input.GetAxisRaw("Horizontal");
        verticalInput = Input.GetAxisRaw("Vertical");

        // Use GetKeyDown for jump to avoid repeated key polling issues
        if (Input.GetKeyDown(jumpKey))
        {
            jumpAttemptCounter++;
            TryJump();
        }

        // Start crouch
        if (Input.GetKeyDown(crouchKey))
        {
            transform.localScale = new Vector3(transform.localScale.x, crouchYScale, transform.localScale.z);
            rb.AddForce(Vector3.down * 5f, ForceMode.Impulse);
            if (debugMode) Debug.Log("[PlayerMovement] Crouch started");
        }

        // Stop crouch
        if (Input.GetKeyUp(crouchKey))
        {
            transform.localScale = new Vector3(transform.localScale.x, startYScale, transform.localScale.z);
            if (debugMode) Debug.Log("[PlayerMovement] Crouch stopped");
        }

        // Sprint detection (log when toggled)
        if (grounded && Input.GetKeyDown(sprintKey))
        {
            if (debugMode) Debug.Log("[PlayerMovement] Sprint key down");
        }
        if (grounded && Input.GetKeyUp(sprintKey))
        {
            if (debugMode) Debug.Log("[PlayerMovement] Sprint key up");
        }
    }

    /// <summary>
    /// Simplified jump logic: checks if jump is available and applies force directly.
    /// No coroutines, no damping overrides, just straightforward impulse.
    /// </summary>
    private void TryJump()
    {
        if (!grounded || jumpCooldownTimer > 0f)
        {
            if (debugMode)
            {
                Debug.Log($"[PlayerMovement] Jump denied: grounded={grounded}, cooldownTimer={jumpCooldownTimer:F3}");
            }
            return;
        }

        // Reset Y velocity for consistent jump behavior
        rb.linearVelocity = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);

        // Apply jump impulse (VelocityChange ignores mass/damping)
        rb.AddForce(Vector3.up * jumpForce, ForceMode.VelocityChange);

        jumpCooldownTimer = jumpCooldown;
        exitingSlope = true;
        jumpSuccessCounter++;

        if (debugMode)
        {
            Debug.Log($"[PlayerMovement] Jump executed! Attempts:{jumpAttemptCounter} Successes:{jumpSuccessCounter}");
        }
    }

    private void StateHandler()
    {
        // Mode - Crouching
        if (Input.GetKey(crouchKey))
        {
            state = MovementState.crouching;
            moveSpeed = crouchSpeed;
        }
        // Mode - Sprinting
        else if (grounded && Input.GetKey(sprintKey))
        {
            state = MovementState.sprinting;
            moveSpeed = sprintSpeed;
        }
        // Mode - Walking
        else if (grounded)
        {
            state = MovementState.walking;
            moveSpeed = walkSpeed;
        }
        // Mode - Air
        else
        {
            state = MovementState.air;
        }
    }

    void MovePlayer()
    {
        // Calculate movement direction
        moveDir = orientation.forward * verticalInput + orientation.right * horizontalInput;

        // Move the player
        // On slope
        if (OnSlope() && !exitingSlope)
        {
            rb.AddForce(GetSlopeMoveDirection() * moveSpeed * 20f, ForceMode.Force);

            if (rb.linearVelocity.y > 0)
                rb.AddForce(Vector3.down * 80f, ForceMode.Force);
        }

        // On ground
        if (grounded)
            rb.AddForce(moveDir.normalized * moveSpeed * 10f, ForceMode.Force);

        // In air
        else if (!grounded)
            rb.AddForce(moveDir.normalized * moveSpeed * 10f * airMultiplier, ForceMode.Force);

        // Turn gravity off while on slope
        rb.useGravity = !OnSlope();

        // Safety cap to avoid exploding velocities (helps stability during recording)
        if (rb.linearVelocity.magnitude > 50f)
        {
            rb.linearVelocity = rb.linearVelocity.normalized * 50f;
            if (debugMode) Debug.LogWarning("[PlayerMovement] Velocity capped to 50 to prevent explosion.");
        }

        if (debugMode)
        {
            // Log applied movement angle and magnitude occasionally
            if (debugTimer >= debugLogInterval - 0.02f) // small timing hack to reduce duplicates
            {
                Debug.Log($"[PlayerMovement] AppliedForce moveDir:{moveDir.normalized} moveSpeed:{moveSpeed} OnSlope:{OnSlope()}");
            }
        }
    }

    void SpeedControl()
    {
        if (OnSlope() && !exitingSlope)
        {
            if (rb.linearVelocity.magnitude > moveSpeed)
                rb.linearVelocity = rb.linearVelocity.normalized * moveSpeed;
        }
        else
        {
            Vector3 flatVel = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);

            // Limit velocity if needed
            if (flatVel.magnitude > moveSpeed)
            {
                Vector3 limitedVel = flatVel.normalized * moveSpeed;
                rb.linearVelocity = new Vector3(limitedVel.x, rb.linearVelocity.y, limitedVel.z);
            }
        }
    }

    private bool OnSlope()
    {
        if (Physics.Raycast(transform.position, Vector3.down, out slopeHit, playerHeight * 0.5f + 0.3f))
        {
            float angle = Vector3.Angle(Vector3.up, slopeHit.normal);
            return angle < maxSlopeAngle && angle != 0;
        }

        return false;
    }

    private Vector3 GetSlopeMoveDirection()
    {
        return Vector3.ProjectOnPlane(moveDir, slopeHit.normal).normalized;
    }

    // Simple on-screen debug overlay to inspect inputs and state during recording
    void OnGUI()
    {
        if (!debugMode) return;

        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.Label($"Grounded: {grounded}");
        GUILayout.Label($"Jump Cooldown: {jumpCooldownTimer:F3}");
        GUILayout.Label($"State: {state}");
        GUILayout.Label($"H Input: {horizontalInput:F2} V Input: {verticalInput:F2}");
        GUILayout.Label($"Rigidbody Vel: {rb.linearVelocity.magnitude:F2}");
        GUILayout.Label($"JumpAttempts: {jumpAttemptCounter} JumpSuccess: {jumpSuccessCounter}");
        GUILayout.EndArea();
    }
}
