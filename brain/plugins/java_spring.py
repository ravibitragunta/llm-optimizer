import re
from brain.plugins.base import ConversationPlugin, Entity

SPRING_CLASSES_REGEX = re.compile(
    r'\b[A-Z][a-zA-Z]+(?:Template|Service|Repository|Controller|Manager|'
    r'Factory|Handler|Mapper|Config|Configuration|Exception|Filter|'
    r'Interceptor|Advisor|Aspect|Bean|Component|Entity)\b'
)
SPRING_ANNOTATIONS_REGEX = re.compile(
    r'@(?:Repository|Service|Controller|RestController|Component|Configuration|'
    r'Bean|Autowired|Transactional|RequestMapping|GetMapping|PostMapping|'
    r'PutMapping|DeleteMapping|PathVariable|RequestBody|ResponseBody|'
    r'SpringBootApplication|Valid|NotNull|NotBlank|NotEmpty|Size|Email|'
    r'Min|Max|ControllerAdvice|ExceptionHandler|ResponseStatus)\b'
)
JDBC_SPECIFIC = [
    "NamedParameterJdbcTemplate", "JdbcTemplate", "SqlParameterSource",
    "MapSqlParameterSource", "BeanPropertySqlParameterSource",
    "RowMapper", "BeanPropertyRowMapper", "ResultSet", "PreparedStatement"
]
SECURITY_SPECIFIC = [
    "SecurityFilterChain", "UserDetailsService", "JwtAuthenticationFilter",
    "AuthenticationManager", "PasswordEncoder", "BCryptPasswordEncoder"
]
SPRING_EXCEPTION_PATTERNS = [
    "GlobalExceptionHandler", "ControllerAdvice", "ExceptionHandler",
    "ResponseEntity", "HttpStatus", "ResponseStatus",
    "RuntimeException", "IllegalArgumentException", "IllegalStateException",
    "MethodArgumentNotValidException", "BindingResult", "FieldError",
    "@Valid", "@NotNull", "@NotBlank", "@NotEmpty", "@Size", "@Email",
    "@Min", "@Max", "@Pattern", "ProblemDetail", "ErrorResponse",
]

class JavaSpringPlugin(ConversationPlugin):

    @property
    def name(self) -> str:
        return "java_spring"

    @property
    def file_extensions(self) -> list:
        return ['.java']

    def extract_entities(self, text: str) -> list:
        entities = []
        seen = set()

        for match in SPRING_CLASSES_REGEX.findall(text):
            if match not in seen:
                entities.append(Entity(match, 'class', 0.9))
                seen.add(match)

        for match in SPRING_ANNOTATIONS_REGEX.findall(text):
            if match not in seen:
                entities.append(Entity(match, 'annotation', 0.95))
                seen.add(match)

        for kw in JDBC_SPECIFIC + SECURITY_SPECIFIC + SPRING_EXCEPTION_PATTERNS:
            if kw in text and kw not in seen:
                entities.append(Entity(kw, 'concept', 0.9))
                seen.add(kw)

        return entities

    def detect_decisions(self, text: str) -> list:
        decisions = []
        decision_markers = [
            "use constructor injection",
            "use NamedParameterJdbcTemplate",
            "use @Transactional",
            "extends RuntimeException",
            "implements UserDetailsService",
        ]
        for marker in decision_markers:
            if marker.lower() in text.lower():
                decisions.append(marker)
        return decisions

    def get_seed_domains(self) -> list:
        return ['spring_jdbc', 'spring_boot', 'spring_security']

    def get_system_prompt(self) -> str:
        return (
            "You are a senior software engineer generating production-quality Java/Spring Boot code.\n\n"
            "SESSION RULES (follow strictly):\n"
            "1. INCREMENTAL BUILD: Do NOT rewrite established classes. Output only NEW code.\n"
            "2. HONOUR DECISIONS: All items in [ESTABLISHED DECISIONS] are final.\n"
            "3. PRECISION: If adding a method, emit only that method.\n"
            "4. NO QUESTIONS: Infer from context and produce the best answer.\n"
            "5. CODE ONLY: No preamble, no meta-commentary.\n"
        )
