using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using System.IO;

public class ILAgent : Agent
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
    // Tidak dipakai — .Add() tidak pernah dipanggil, selalu kosong
    //private HashSet<Vector2Int> visitedGrids = new HashSet<Vector2Int>();

    private HashSet<Vector2Int> goldenPathGrids = new HashSet<Vector2Int>();
    private HashSet<Vector2Int> explorationGrids = new HashSet<Vector2Int>();

    [Header("Coverage Export")]
    public string coverageFileName = "CoverageData_IL.csv";

    [Header("Bug Reporting")]
    public string fileName = "BugReport_IL.csv";
    public float stuckTimeout = 5f;
    private float stuckTimer = 0f;
    private Vector3 lastPosition;

    [Header("Statistics")]
    private int jumpCount, sprintCount, totalActions;
    private int episodeCount = 0;
    //private int totalGoldenPathGrids = 0;
    //private int totalExplorationGrids = 0;

    [Header("Map References")]
    public MapOrganizer mapOrganizer;

    [Header("Auto Golden Path Recording")]
    public string rawPathFileName = "GoldenPath/GoldenPath_Raw.txt";
    private HashSet<Vector2Int> rawPathGrids = new HashSet<Vector2Int>();

    private Transform spawnPoint;

    [Header("Debug System")]
    public bool enableDebugLog = true;
    public bool enableVisualDebug = true;

    [Header("Physics Detection")]
    public float physicsVelocityThreshold = 300f;
    public float physicsVelocityDuration = 0.1f; // harus melampaui threshold selama 0.1 detik
    private float physicsVelocityTimer = 0f;

    [Header("Position Logger")]
    [Tooltip("Nama file CSV untuk log posisi spasial setiap 5 frame.")]
    public string positionFileName = "PositionLog_IL_MapB.csv";
    [Tooltip("Catat posisi setiap N frame. Default 5 sesuai metodologi skripsi.")]
    public int positionLogInterval = 5;
    private StreamWriter positionWriter;
    private int frameCounter = 0;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        rb.freezeRotation = true;
        spawnPoint = GameObject.Find("SpawnPoint")?.transform;
        InitPositionLogger();
    }
    private void Update()
    {
        // Menekan tombol Enter (Return) untuk menyimpan data grid sebelum mematikan Play mode
        if (Input.GetKeyDown(KeyCode.Return))
        {
            SaveRawPathToFile();
        }

        // Catat posisi setiap positionLogInterval frame
        frameCounter++;
        if (frameCounter % positionLogInterval == 0)
        {
            LogPosition();
        }
    }

    void OnDestroy()
    {
        ClosePositionLogger();
    }

    //private void FixedUpdate()
    //{
    //    // Memaksa sensor untuk terus menyala dan mencatat grid selama mesin fisika Unity berjalan, terlepas dari status ML-Agents
    //    TrackGridCoverage();
    //}

    // -----------------------------------------------------------------------
    // EPISODE
    // -----------------------------------------------------------------------
    public override void OnEpisodeBegin()
    {
        episodeCount++;

        // Statistik Aksi untuk TensorBoard
        if (totalActions > 0)
        {
            Academy.Instance.StatsRecorder.Add("Actions/JumpUsage", (float)jumpCount / totalActions);
            Academy.Instance.StatsRecorder.Add("Actions/SprintUsage", (float)sprintCount / totalActions);
        }

        int goldenPathCount = goldenPathGrids.Count;
        int explorationCount = explorationGrids.Count;
        float totalCoverage = goldenPathCount + explorationCount;

        // Kirim data Coverage terakhir untuk dievaluasi di TensorBoard lintas model
        if (totalCoverage > 0)
        {
            Academy.Instance.StatsRecorder.Add("Grid_Exploration/Golden_Path_Count", goldenPathCount);
            Academy.Instance.StatsRecorder.Add("Grid_Exploration/New_Area_Count", explorationCount);
            Academy.Instance.StatsRecorder.Add("Grid_Exploration/Total_Coverage", totalCoverage);

            float explorationPercentage = ((float)explorationCount / totalCoverage) * 100f;
            Academy.Instance.StatsRecorder.Add("Exploration_Ratio/New_Area_Percentage", explorationPercentage);
        }

        SaveCoverageToFile();

        // Reset state
        //visitedGrids.Clear();
        //visitedGrids.TrimExcess();
        goldenPathGrids.Clear();
        explorationGrids.Clear();
        goldenPathGrids.TrimExcess();
        explorationGrids.TrimExcess();

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

    // -----------------------------------------------------------------------
    // OBSERVATIONS & ACTIONS
    // -----------------------------------------------------------------------
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
        totalActions++;
        DetectAnomalies();
        ProcessMovement(actions);
        TrackGridCoverage();
    }

    // -----------------------------------------------------------------------
    // MOVEMENT
    // -----------------------------------------------------------------------
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

        // Mengambil kecepatan X dan Z saat ini tanpa menyentuh kecepatan jatuh (Y)
        Vector3 flatVel = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);

        // Jika kecepatan melampaui batas wajar, potong dan normalkan kembali
        if (flatVel.magnitude > speed)
        {
            Vector3 limitedVel = flatVel.normalized * speed;
            rb.linearVelocity = new Vector3(limitedVel.x, rb.linearVelocity.y, limitedVel.z);
        }

        grounded = Physics.Raycast(transform.position, Vector3.down, playerHeight * 0.5f + 0.75f, whatIsGround)
                   && !FallZoneTrigger.IsInFallZone(GetComponent<Collider>());
        //if (enableDebugLog)
        //    Debug.Log($"Raycast length: {playerHeight * 0.5f + 0.75f}, " +
        //    $"Position Y: {transform.position.y}, " +
        //    $"Hit: {Physics.Raycast(transform.position, Vector3.down, playerHeight * 0.5f + 0.2f, whatIsGround)}");
        rb.linearDamping = grounded ? groundDrag : 0f;

        float forceMult = grounded ? 10f : 10f * airMultiplier;
        rb.AddForce(moveDir.normalized * speed * forceMult, ForceMode.Force);

        // Lompat
        if (isJumping && readyToJump && grounded)
        {
            //if (enableDebugLog)
            //    Debug.Log($"isJumping: {isJumping}, readyToJump: {readyToJump}, grounded: {grounded}");
            jumpCount++;
            readyToJump = false;
            rb.linearVelocity = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);
            rb.AddForce(transform.up * jumpForce, ForceMode.Impulse);
            Invoke(nameof(ResetJump), jumpCooldown);
        }
    }

    // -----------------------------------------------------------------------
    // COVERAGE
    // -----------------------------------------------------------------------
    // Grid validity khusus untuk ILAgent pakai raycast, karena asumsi ahli adalah ground truth
    // Grid validity berdasarkan navmesh dipindah ke MapOrganizer
    private void TrackGridCoverage()
    {
        int gx = Mathf.FloorToInt(transform.position.x / gridCellSize);
        int gz = Mathf.FloorToInt(transform.position.z / gridCellSize);
        Vector2Int currentGrid = new Vector2Int(gx, gz);

        //// Angkat titik awal Raycast 1 unit ke atas (setinggi dada karakter)
        //Vector3 rayStart = transform.position + (Vector3.up * 1.0f);
        //float rayDistance = 10.0f;

        //// Debugging Visual: Sinar berwarna Magenta akan muncul di Scene view agar Anda bisa melihat sensornya
        //Debug.DrawRay(rayStart, Vector3.down * rayDistance, Color.magenta);

        ////Debug.Log("Sensor grid sedang aktif bekerja.");

        //// Jika sinar menabrak lantai dengan Layer yang tepat, catat grid-nya
        //if (Physics.Raycast(rayStart, Vector3.down, rayDistance, whatIsGround))
        //{
        //    rawPathGrids.Add(currentGrid);
        //}
        if (mapOrganizer != null && mapOrganizer.IsValidNavMeshGrid(currentGrid))
        {
            if (mapOrganizer.IsGoldenPathArea(currentGrid))
            {
                if (goldenPathGrids.Add(currentGrid))
                {
                    //AddReward(0.1f);
                    if (enableDebugLog)
                        Debug.Log($"Agen menemukan Grid Golden Path baru: {currentGrid}");
                }
            }
            else
            {
                if (explorationGrids.Add(currentGrid))
                {
                    //AddReward(0.3f);
                    if (enableDebugLog)
                        Debug.Log($"<Agen mengeksplorasi area bebas di Grid: {currentGrid}");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // ANOMALY DETECTION
    // -----------------------------------------------------------------------
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
            ReportBug("NavigationBug_Stuck");
            if (enableDebugLog)
                Debug.LogError($"Navigation Bug! Agen stuck di posisi {transform.position}. Episode Berakhir.");
            EndEpisode();
        }

        // Fall Detection
        if (transform.position.y < -10f)
        {
            ReportBug("GeometryBug_FallThroughFloor");
            if (enableDebugLog)
                Debug.LogError($"Geometry Bug! Agen jatuh di koordinat {transform.position.x}, {transform.position.y}, {transform.position.z}. Episode Berakhir.");
            EndEpisode();
        }

        // Physics Explosion
        if (rb.linearVelocity.magnitude > physicsVelocityThreshold)
        {
            physicsVelocityTimer += Time.fixedDeltaTime;
            if (physicsVelocityTimer >= physicsVelocityDuration)
            {
                ReportBug("PhysicsBug_VelocityExplosion");
                if (enableDebugLog)
                {
                    Debug.LogError($"Physics Bug! Ledakan kecepatan terjadi. Episode Berakhir.");
                    Debug.Log($"Current velocity magnitude: {rb.linearVelocity.magnitude}");
                }
                EndEpisode();
                physicsVelocityTimer = 0f;
            }
            //ReportBug("Physics_VelocityExplosion");
            //if (enableDebugLog)
                
            //rb.linearVelocity = Vector3.zero;
            //EndEpisode();
        }
        else
        {
            physicsVelocityTimer = 0f;
        }
    }

    private void ReportBug(string type)
    {
        try
        {
            string path = Path.Combine(Application.dataPath, fileName);
            string row = $"{System.DateTime.Now:HH:mm:ss},{type},{transform.position.x:F2},{transform.position.y:F2},{transform.position.z:F2},{CompletedEpisodes}\n";
            if (!File.Exists(path)) File.WriteAllText(path, "Time,Type,X,Y,Z,Episode\n");
            File.AppendAllText(path, row);
        }
        catch { /* Mengabaikan file lock */ }
    }

    // Metode Heuristic diperlukan untuk Demonstration Recorder untuk merekam pergerakan manusia
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        //Debug.Log("Heuristic masih merespons keyboard");

        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;

        // Inisialisasi default ke status diam (0)
        discreteActions[0] = 0;
        discreteActions[1] = 0;
        discreteActions[2] = 0;
        discreteActions[3] = 0;

        // Pemetaan Input Keyboard ke Discrete Actions
        if (Input.GetKey(KeyCode.W)) discreteActions[0] = 1; // Maju
        else if (Input.GetKey(KeyCode.S)) discreteActions[0] = 2; // Mundur

        float mouseX = Input.GetAxis("Mouse X");

        if (Input.GetKey(KeyCode.D) || mouseX > 0.1f) discreteActions[1] = 1; // Putar Kanan
        else if (Input.GetKey(KeyCode.A) || mouseX < -0.1f) discreteActions[1] = 2; // Putar Kiri

        if (Input.GetKey(KeyCode.LeftShift)) discreteActions[2] = 1; // Sprint aktif

        
        if (Input.GetKey(KeyCode.Space)) discreteActions[3] = 1; // Lompat aktif
    }

    private void ResetJump() => readyToJump = true;

    public void SaveRawPathToFile()
    {
        string path = Path.Combine(Application.dataPath, rawPathFileName);

        using (StreamWriter writer = new StreamWriter(path, true))
        {
            foreach (var grid in rawPathGrids)
            {
                writer.WriteLine($"{grid.x},{grid.y}");
            }
        }

        Debug.Log($"<color=green>SUCCESS:</color> Sesi rekaman ini berhasil ditambahkan! {rawPathGrids.Count} grid disuntikkan ke file.");

        // Bersihkan memori agen saat ini agar siap jika tidak mereset Play Mode
        rawPathGrids.Clear();
    }

    private void SaveCoverageToFile()
    {
        if (goldenPathGrids.Count == 0 && explorationGrids.Count == 0) return;

        string path = Path.Combine(Application.dataPath, coverageFileName);
        bool isNewFile = !File.Exists(path);

        using (StreamWriter writer = new StreamWriter(path, true))
        {
            if (isNewFile)
            {
                writer.WriteLine("Episode,X,Z,AreaType");
            }

            foreach (var grid in goldenPathGrids)
            {
                writer.WriteLine($"{episodeCount},{grid.x},{grid.y},GoldenPath");
            }

            foreach (var grid in explorationGrids)
            {
                writer.WriteLine($"{episodeCount},{grid.x},{grid.y},Exploration");
            }
        }
    }

    private void OnDrawGizmos()
    {
        if (!Application.isPlaying || !enableVisualDebug) return;

        if (goldenPathGrids != null)
        {
            Gizmos.color = new Color(0f, 1f, 0f, 0.4f);
            foreach (var grid in goldenPathGrids)
            {
                Vector3 pos = new Vector3(grid.x * gridCellSize, 0.1f, grid.y * gridCellSize);
                Gizmos.DrawCube(pos, new Vector3(gridCellSize, 0.1f, gridCellSize));
            }
        }

        if (explorationGrids != null)
        {
            Gizmos.color = new Color(0f, 1f, 1f, 0.4f);
            foreach (var grid in explorationGrids)
            {
                Vector3 pos = new Vector3(grid.x * gridCellSize, 0.1f, grid.y * gridCellSize);
                Gizmos.DrawCube(pos, new Vector3(gridCellSize, 0.1f, gridCellSize));
            }
        }
    }

    [ContextMenu("Hapus Data Perekaman Lama")]
    public void ClearOldPathData()
    {
        string path = Path.Combine(Application.dataPath, rawPathFileName);
        if (File.Exists(path))
        {
            File.Delete(path);
            Debug.Log($"<color=yellow>RESET:</color> File {rawPathFileName} berhasil dihapus. Anda siap merekam data baru dari nol.");
        }
        else
        {
            Debug.Log("File belum ada, Anda bisa langsung mulai merekam.");
        }
    }

    // -----------------------------------------------------------------------
    // POSITION LOGGER
    // -----------------------------------------------------------------------

    private void InitPositionLogger()
    {
        try
        {
            string path = Path.Combine(Application.dataPath, positionFileName);
            bool isNewFile = !File.Exists(path);
            positionWriter = new StreamWriter(path, append: true);
            if (isNewFile)
                positionWriter.WriteLine("Episode,Frame,X,Y,Z");
            positionWriter.AutoFlush = true;
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[CDRLAgent] Gagal membuka position logger: {e.Message}");
        }
    }

    private void LogPosition()
    {
        if (positionWriter == null) return;
        try
        {
            Vector3 pos = transform.position;
            positionWriter.WriteLine(
                $"{CompletedEpisodes},{frameCounter},{pos.x:F2},{pos.y:F2},{pos.z:F2}");
        }
        catch { /* Ignore file lock */ }
    }

    private void ClosePositionLogger()
    {
        try
        {
            positionWriter?.Flush();
            positionWriter?.Close();
            positionWriter = null;
        }
        catch { /* Ignore */ }
    }
}