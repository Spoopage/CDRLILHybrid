using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using static UnityEngine.GraphicsBuffer;
using System.IO;

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

    [Header("Coverage & Reward System")]
    public float gridCellSize = 1f;
    public float rewardPerNewCell = 0.1f;
    // Area yang sudah dikunjungi dalam episode ini
    private HashSet<Vector2Int> visitedGrids = new HashSet<Vector2Int>();

    [Header("Action Tracking")]
    private int jumpCount = 0;
    private int sprintCount = 0;
    private int totalActions = 0;

    [Header("Bug Reporting")]
    public string fileName = "BugReport.csv";

    [Header("Stuck Detection")]
    public float stuckTimeout = 5f; // Berapa detik agen boleh diam/nyangkut
    private float stuckTimer = 0f;
    private Vector3 lastPosition;

    [Header("Golden Path Settings")]
    public List<Transform> goldenPathWaypoints; // Masukkan list titik jalan di Inspector
    public float maxDeviationDistance = 10f; // Jarak maksimal sebelum dianggap Keluar Jalur

    private Rigidbody rb;
    private Vector3 moveDir;
    private float currentSpeed;
    public Transform target;
    [System.NonSerialized]
    public static int totalFalls = 0;
    private int episodeCount = 0;
    private LayerMask obstacleLayer;
    private LayerMask buildingLayer;

    [Header("Spawn Settings")]
    public Transform spawnPoint;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        rb.freezeRotation = true;

        if (spawnPoint == null)
            spawnPoint = GameObject.Find("SpawnPoint")?.transform;
        
        #if UNITY_EDITOR
        // Reset static counters in editor when not doing domain reload
        if (!UnityEditor.EditorApplication.isPlaying)
            totalFalls = 0;
        #endif
    }

    public override void OnEpisodeBegin()
    {
        // Memory management: Bersihkan memori yang tidak terpakai setiap episode untuk mencegah penumpukan data
        Resources.UnloadUnusedAssets();
        System.GC.Collect();
        episodeCount++;

        if (totalActions > 0)
        {
            // Menghitung rasio/persentase penggunaan aksi
            float jumpRatio = (float)jumpCount / totalActions;
            float sprintRatio = (float)sprintCount / totalActions;

            Academy.Instance.StatsRecorder.Add("Actions/JumpUsage", jumpRatio);
            Academy.Instance.StatsRecorder.Add("Actions/SprintUsage", sprintRatio);
        }

        // Reset counter untuk episode baru
        jumpCount = 0;
        sprintCount = 0;
        totalActions = 0;

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
        // Posisi Lokal Agen (3 data: x, y, z)
        // Agar agen tahu dia sudah di area mana
        sensor.AddObservation(transform.localPosition);

        // Arah Hadap Agen (3 data: Forward Vector)
        // Penting agar agen tahu orientasi tubuhnya terhadap ruang
        sensor.AddObservation(transform.forward);

        // Kecepatan Linear Lengkap (3 data: Vector3, bukan cuma magnitude)
        // Mengetahui arah gerak saat ini membantu agen menyadari kalau dia cuma berputar-putar
        sensor.AddObservation(rb.linearVelocity);

        // Kecepatan Sudut (1 data: Angular Velocity di sumbu Y)
        // Membantu agen menyadari kalau dia sedang berputar (spinning)
        sensor.AddObservation(rb.angularVelocity.y);

        // Status Grounded (1 data)
        sensor.AddObservation(grounded ? 1.0f : 0.0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        totalActions++;
        // Cek Sprint
        if (actions.DiscreteActions[2] == 1) sprintCount++;

        // Cek Jump
        if (actions.DiscreteActions[3] == 1 && grounded) jumpCount++;

        // Discrete Actions
        int moveZ = actions.DiscreteActions[0];   // 0: Diam, 1: Maju, 2: Mundur
        int rotateY = actions.DiscreteActions[1]; // 0: Diam, 1: Kanan, 2: Kiri
        int isSprinting = actions.DiscreteActions[2]; // 0: Jalan, 1: Sprint
        int isJumping = actions.DiscreteActions[3];   // 0: Tidak, 1: Lompat

        // Logic Rotasi
        float rotationInput = 0f;
        if (rotateY == 1) rotationInput = 1f;
        if (rotateY == 2) rotationInput = -1f;
        transform.Rotate(Vector3.up, rotationInput * rotationSpeed * Time.fixedDeltaTime);

        // Logic Movement & Sprint
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

        // Logic Jumping
        if (isJumping == 1 && readyToJump && grounded)
        {
            readyToJump = false;
            rb.linearVelocity = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);
            rb.AddForce(transform.up * jumpForce, ForceMode.Impulse);
            Invoke(nameof(ResetJump), jumpCooldown);
        }

        // Reward System (Eksplorasi Coverage)
        TrackCoverageAndReward();

        //DetectPhysicsBugs();
        //CheckGoldenPathDeviation();

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
            //ReportBug("StuckInGeometry"); // Log bug stuck
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

    private void DetectPhysicsBugs()
    {
        // Deteksi Clipping (Overlap dengan Building)
        Collider[] overlap = Physics.OverlapSphere(transform.position, 0.3f, buildingLayer);
        if (overlap.Length > 0)
        {
            ReportBug("Physics_ClippingDetected");
        }

        // Deteksi Kecepatan Abnormal (Velocity Explosion)
        if (rb.linearVelocity.magnitude > sprintSpeed * 10f)
        {
            ReportBug("Physics_VelocityExplosion");
        }
    }

    private void CheckGoldenPathDeviation()
    {
        if (goldenPathWaypoints == null || goldenPathWaypoints.Count == 0) return;

        float minDistance = float.MaxValue;
        foreach (Transform point in goldenPathWaypoints)
        {
            float dist = Vector3.Distance(transform.position, point.position);
            if (dist < minDistance) minDistance = dist;
        }

        if (minDistance > maxDeviationDistance)
        {
            // Jangan langsung EndEpisode, cukup lapor sebagai observasi
            ReportBug("Navigation_GoldenPathDeviation");
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

    // Bug reporting system: Mencatat jenis bug, posisi, dan waktu ke file CSV
    private void ReportBug(string bugType)
    {
        try
        {
            Vector3 bugPos = transform.position;
            string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            string csvRow = $"{timestamp},{bugType},{bugPos.x:F2},{bugPos.y:F2},{bugPos.z:F2},{CompletedEpisodes}";

            string path = Path.Combine(Application.dataPath, fileName);

            // Jika file tidak ada, buat header-nya
            if (!File.Exists(path))
            {
                File.WriteAllText(path, "Timestamp,BugType,PosX,PosY,PosZ,Episode\n");
            }

            // Gunakan FileStream dengan FileShare agar lebih toleran terhadap akses bersamaan
            using (StreamWriter sw = new StreamWriter(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite)))
            {
                sw.WriteLine(csvRow);
            }

            Debug.LogWarning($"<color=red>BUG DETECTED:</color> {bugType} at {bugPos}");
        }
        catch (IOException e)
        {
            // Jika file terkunci, kita log ke console saja agar training tidak stop
            Debug.LogWarning("Gagal menulis ke CSV karena file sedang digunakan: " + e.Message);
        }
    }

    // Agar agen tidak sering menabrak perimeter
    private void OnCollisionEnter(Collision collision)
    {
        if (/*collision.gameObject.CompareTag("Obstacle") ||*/ collision.gameObject.layer == LayerMask.NameToLayer("Perimeter"))
        {
            AddReward(-0.01f); // Penalti kecil agar agen tidak sering menabrak
        }
    }

    private void ResetJump()
    {
        readyToJump = true;
    }

    // Fungsi deteksi fall bug
    private void Update()
    {
        if (transform.position.y < -10f)
        {
            totalFalls++;

            // Log posisi terakhir
            ReportBug("FallThroughFloor");

            // Beri penalti besar jika jatuh ke luar map, lalu akhiri episode
            SetReward(-1f);
            EndEpisode();
        }
    }
}