using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using System.IO;

public class CDRLAgentOG : Agent
{
    [Header("Movement Settings")]
    public float walkSpeed = 5f;
    public float sprintSpeed = 8f;
    public float rotationSpeed = 150f;
    public float groundDrag = 5f;

    [Header("Jumping")]
    public float jumpForce = 7f;
    public float jumpCooldown = 0.25f;
    public float airMultiplier = 0.4f;
    private bool readyToJump = true;

    [Header("Physics & Environment")]
    public float playerHeight = 2f;
    public LayerMask whatIsGround;
    public LayerMask obstacleLayer;
    private bool grounded;
    private Rigidbody rb;

    [Header("Coverage Tracking")]
    public float gridCellSize = 1f;
    private HashSet<Vector2Int> visitedGrids = new HashSet<Vector2Int>();

    [Header("Bug Reporting")]
    public string fileName = "BugReport_CDRL.csv";
    public float stuckTimeout = 5f;
    private float stuckTimer = 0f;
    private Vector3 lastPosition;

    [Header("Statistics")]
    private int jumpCount, sprintCount, totalActions;
    private int episodeCount = 0;

    private Transform spawnPoint;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        rb.freezeRotation = true;
        spawnPoint = GameObject.Find("SpawnPoint")?.transform;
    }

    public override void OnEpisodeBegin()
    {
        // Memory management: Bersihkan memori yang tidak terpakai setiap episode untuk mencegah penumpukan data
        //Resources.UnloadUnusedAssets();
        //System.GC.Collect();

        episodeCount++;

        // Statistik Aksi untuk TensorBoard
        if (totalActions > 0)
        {
            Academy.Instance.StatsRecorder.Add("Actions/JumpUsage", (float)jumpCount / totalActions);
            Academy.Instance.StatsRecorder.Add("Actions/SprintUsage", (float)sprintCount / totalActions);
        }

        // Kirim data Coverage terakhir sebelum reset
        if (visitedGrids.Count > 0)
            Academy.Instance.StatsRecorder.Add("Custom/CoverageCount", visitedGrids.Count);

        // Reset state
        visitedGrids.Clear();
        visitedGrids.TrimExcess();
        jumpCount = 0; sprintCount = 0; totalActions = 0;
        stuckTimer = 0f;

        // Relokasi ke Spawn Point
        if (spawnPoint)
        {
            transform.position = spawnPoint.position;
            transform.rotation = spawnPoint.rotation;
        }

        rb.linearVelocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.forward);
        sensor.AddObservation(rb.linearVelocity);
        sensor.AddObservation(rb.angularVelocity.y);
        sensor.AddObservation(grounded ? 1.0f : 0.0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        //Debug.Log("Agen sedang menerima aksi: " + actions.DiscreteActions[0]);
        totalActions++;
        ProcessMovement(actions);
        TrackGridCoverage();
        DetectAnomalies();
    }

    private void ProcessMovement(ActionBuffers actions)
    {
        int moveZ = actions.DiscreteActions[0];
        int rotateY = actions.DiscreteActions[1];
        bool isSprinting = actions.DiscreteActions[2] == 1;
        bool isJumping = actions.DiscreteActions[3] == 1;

        if (isSprinting) sprintCount++;

        // Rotasi
        float rotDir = rotateY == 1 ? 1f : (rotateY == 2 ? -1f : 0f);
        transform.Rotate(Vector3.up, rotDir * rotationSpeed * Time.fixedDeltaTime);

        // Gerak
        float moveInput = moveZ == 1 ? 1f : (moveZ == 2 ? -1f : 0f);
        float speed = isSprinting ? sprintSpeed : walkSpeed;
        Vector3 moveDir = transform.forward * moveInput;

        grounded = Physics.Raycast(transform.position, Vector3.down, playerHeight * 0.5f + 0.2f, whatIsGround);
        rb.linearDamping = grounded ? groundDrag : 0f;

        float forceMult = grounded ? 10f : 10f * airMultiplier;
        rb.AddForce(moveDir.normalized * speed * forceMult, ForceMode.Force);

        // Lompat
        if (isJumping && readyToJump && grounded)
        {
            jumpCount++;
            readyToJump = false;
            rb.linearVelocity = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);
            rb.AddForce(transform.up * jumpForce, ForceMode.Impulse);
            Invoke(nameof(ResetJump), jumpCooldown);
        }
    }

    private void TrackGridCoverage()
    {
        int gx = Mathf.FloorToInt(transform.position.x / gridCellSize);
        int gz = Mathf.FloorToInt(transform.position.z / gridCellSize);
        if (visitedGrids.Add(new Vector2Int(gx, gz)))
        {
            AddReward(0.1f);
        }
    }

    private void DetectAnomalies()
    {
        // Stuck Detection
        if (Vector3.Distance(transform.position, lastPosition) < 0.2f)
            stuckTimer += Time.fixedDeltaTime;
        else
        {
            stuckTimer = 0f;
            lastPosition = transform.position;
        }

        if (stuckTimer > stuckTimeout)
        {
            ReportBug("StuckInGeometry");
            EndEpisode();
        }

        // Fall Detection
        if (transform.position.y < -10f)
        {
            ReportBug("FallThroughFloor");
            EndEpisode();
        }

        // Physics Explosion
        if (rb.linearVelocity.magnitude > sprintSpeed * 10f)
        {
            ReportBug("Physics_VelocityExplosion");
            rb.linearVelocity = Vector3.zero;
            EndEpisode();
        }
    }

    private void ReportBug(string type)
    {
        try
        {
            string path = Path.Combine(Application.dataPath, fileName);
            string row = $"{System.DateTime.Now:HH:mm:ss},{type},{transform.position.x:F2},{transform.position.z:F2},{CompletedEpisodes}\n";
            if (!File.Exists(path)) File.WriteAllText(path, "Time,Type,X,Z,Episode\n");
            File.AppendAllText(path, row);
        }
        catch { /* Ignore lock */ }
    }

    private void ResetJump() => readyToJump = true;
}