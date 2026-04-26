using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using static UnityEngine.GraphicsBuffer;

public class RLAgent : Agent
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

    [Header("Ground Check")]
    public float playerHeight = 2f;
    public LayerMask whatIsGround;
    private bool grounded;

    [Header("Stuck Detection")]
    public float stuckTimeout = 5f; // Berapa detik agen boleh diam/nyangkut
    private float stuckTimer = 0f;
    private Vector3 lastPosition;

    [Header("Coverage & Reward System")]
    public float gridCellSize = 1f;
    public float rewardPerNewCell = 0.1f;
    // Area yang sudah dikunjungi dalam episode ini
    private HashSet<Vector2Int> visitedGrids = new HashSet<Vector2Int>();

    private Rigidbody rb;
    private Vector3 moveDir;
    private float currentSpeed;
    public Transform target;
    public static int totalFalls = 0;
    private int episodeCount = 0;

    [Header("Spawn Settings")]
    public Transform spawnPoint; // Tarik objek SpawnPoint ke sini di Inspector

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        rb.freezeRotation = true;

        if (spawnPoint == null)
            spawnPoint = GameObject.Find("SpawnPoint")?.transform;
    }

    public override void OnEpisodeBegin()
    {
        Resources.UnloadUnusedAssets();
        System.GC.Collect();
        episodeCount++;

        if (episodeCount % 10 == 0 && visitedGrids.Count > 0)
    {
        Academy.Instance.StatsRecorder.Add("Custom/CoverageCount", visitedGrids.Count);
    }

        // Reset Posisi dan Physics
        if (spawnPoint != null)
        {
            transform.position = spawnPoint.position;
            transform.rotation = spawnPoint.rotation;
        }

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // Kirim statistik coverage ke StatsRecorder untuk analisis setelah episode selesai
        if (visitedGrids.Count > 0)
        {
            Academy.Instance.StatsRecorder.Add("Custom/CoverageCount", visitedGrids.Count);
        }

        // Reset Memori Navigasi
        visitedGrids.Clear();
        // Jika HashSet terlalu besar, bersihkan kapasitasnya
        visitedGrids.TrimExcess();
        readyToJump = true;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // RayPerceptionSensor3D sudah memberikan observasi ruang secara otomatis.
        // Di sini kita hanya perlu menambahkan observasi status tubuh agen.

        // 1. Posisi Lokal Agen (3 data: x, y, z)
        // Agar agen tahu dia sudah di area mana (relatif terhadap titik pusat map)
        sensor.AddObservation(transform.localPosition);

        // 2. Arah Hadap Agen (3 data: Forward Vector)
        // Penting agar agen tahu orientasi tubuhnya terhadap ruang
        sensor.AddObservation(transform.forward);

        // 3. Kecepatan Linear Lengkap (3 data: Vector3, bukan cuma magnitude)
        // Mengetahui arah gerak saat ini membantu agen menyadari kalau dia cuma mutar-mutar
        sensor.AddObservation(rb.linearVelocity);

        // 4. Kecepatan Sudut (1 data: Angular Velocity di sumbu Y)
        // Membantu agen menyadari kalau dia sedang berputar (spinning)
        sensor.AddObservation(rb.angularVelocity.y);

        // 5. Status Grounded (1 data)
        sensor.AddObservation(grounded ? 1.0f : 0.0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // 1. Discrete Actions
        int moveZ = actions.DiscreteActions[0];   // 0: Diam, 1: Maju, 2: Mundur
        int rotateY = actions.DiscreteActions[1]; // 0: Diam, 1: Kanan, 2: Kiri
        int isSprinting = actions.DiscreteActions[2]; // 0: Jalan, 1: Sprint
        int isJumping = actions.DiscreteActions[3];   // 0: Tidak, 1: Lompat

        // 2. Logic Rotasi
        float rotationInput = 0f;
        if (rotateY == 1) rotationInput = 1f;
        if (rotateY == 2) rotationInput = -1f;
        transform.Rotate(Vector3.up, rotationInput * rotationSpeed * Time.fixedDeltaTime);

        // 3. Logic Movement & Sprint
        float moveInput = 0f;
        if (moveZ == 1) moveInput = 1f;
        if (moveZ == 2) moveInput = -1f;

        currentSpeed = (isSprinting == 1) ? sprintSpeed : walkSpeed;
        moveDir = transform.forward * moveInput;

        // Cek apakah di tanah
        grounded = Physics.Raycast(transform.position, Vector3.down, playerHeight * 0.5f + 0.2f, whatIsGround);
        rb.linearDamping = grounded ? groundDrag : 0f;

        // Aplikasikan gaya gerak
        if (grounded)
            rb.AddForce(moveDir.normalized * currentSpeed * 10f, ForceMode.Force);
        else
            rb.AddForce(moveDir.normalized * currentSpeed * 10f * airMultiplier, ForceMode.Force);

        // 4. Logic Jumping
        if (isJumping == 1 && readyToJump && grounded)
        {
            readyToJump = false;
            rb.linearVelocity = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);
            rb.AddForce(transform.up * jumpForce, ForceMode.Impulse);
            Invoke(nameof(ResetJump), jumpCooldown);
        }

        // 5. Reward System (Eksplorasi Coverage)
        TrackCoverageAndReward();

        // Stuck Detection
        if (Vector3.Distance(transform.position, lastPosition) < 0.5f)
        {
            stuckTimer += Time.fixedDeltaTime;
        }
        else
        {
            // Reset timer jika agen berhasil bergerak
            stuckTimer = 0f;
            lastPosition = transform.position;
        }

        // Jika timer melebihi batas, berikan hukuman dan reset
        if (stuckTimer > stuckTimeout)
        {
            AddReward(-0.5f); // Berikan penalti karena membuang-buang waktu
            EndEpisode();     // Paksa mulai ulang episode
        }

        // Penalti kecil setiap step agar agen bergerak seefisien mungkin
        // AddReward(-1f / MaxStep); 
    }

    private void TrackCoverageAndReward()
    {
        // Mengubah koordinat dunia nyata menjadi koordinat matriks grid 1x1
        int gridX = Mathf.FloorToInt(transform.position.x / gridCellSize);
        int gridZ = Mathf.FloorToInt(transform.position.z / gridCellSize);
        Vector2Int currentGrid = new Vector2Int(gridX, gridZ);

        // Jika grid ini belum pernah dikunjungi di episode ini
        if (visitedGrids.Add(currentGrid))
        {
            // Berikan imbalan positif karena berhasil menemukan area baru
            AddReward(rewardPerNewCell);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Untuk kontrol AI Player secara manual menggunakan keyboard
        // Untuk fase Perekaman Data Demonstrasi
        var discreteActions = actionsOut.DiscreteActions;

        // Move Z
        if (Input.GetKey(KeyCode.W)) discreteActions[0] = 1;
        else if (Input.GetKey(KeyCode.S)) discreteActions[0] = 2;
        else discreteActions[0] = 0;

        // Rotate Y
        if (Input.GetKey(KeyCode.D)) discreteActions[1] = 1;
        else if (Input.GetKey(KeyCode.A)) discreteActions[1] = 2;
        else discreteActions[1] = 0;

        // Sprint
        discreteActions[2] = Input.GetKey(KeyCode.LeftShift) ? 1 : 0;

        // Jump
        discreteActions[3] = Input.GetKey(KeyCode.Space) ? 1 : 0;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Obstacle") || collision.gameObject.layer == LayerMask.NameToLayer("Perimeter"))
        {
            AddReward(-0.01f); // Penalti kecil agar agen tidak sering menabrak
        }
    }

    private void ResetJump()
    {
        readyToJump = true;
    }

    // Fungsi deteksi anomali (Geometry Bug Y < -10)
    private void Update()
    {
        if (transform.position.y < -10f)
        {
            totalFalls++;
            Academy.Instance.StatsRecorder.Add("Custom/TotalFalls", totalFalls);

            // Beri penalti besar jika jatuh ke luar map, lalu akhiri episode
            SetReward(-1f);
            EndEpisode();
        }
    }
}